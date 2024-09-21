import numpy as np
import sympy as sp
import re
import inspect
import numbers

FUNCTION_START = 0xE000
CONSTANT_START = 0xE100
IMPL_MULT_PATTERN = re.compile(
    fr"(?<=[\da-zA-Z)\u{CONSTANT_START:X}-\uE1FF])([(a-zA-Z_\u{FUNCTION_START:X}-\uEFFF])"
)
DERIVATIVE_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)('+)([a-zA-Z_]*)?\(")

def fix_expr(expr, functions, constants):
    ordered_functions = sorted(functions, key=len, reverse=True)
    ordered_constants = sorted(constants, key=len, reverse=True)
    new_expr = expr.replace("^", "**")
    for index, name in enumerate(ordered_functions):
        placeholder = chr(FUNCTION_START + index)
        new_expr = new_expr.replace(name, placeholder)
    for index, name in enumerate(ordered_constants):
        placeholder = chr(CONSTANT_START + index)
        new_expr = new_expr.replace(name, placeholder)
    new_expr = IMPL_MULT_PATTERN.sub(r"*\1", new_expr)
    for index, name in enumerate(ordered_constants):
        placeholder = chr(CONSTANT_START + index)
        new_expr = new_expr.replace(placeholder, name)
    for index, name in enumerate(ordered_functions):
        placeholder = chr(FUNCTION_START + index)
        new_expr = new_expr.replace(placeholder, name)
    return new_expr

def rename_derivatives(expr, symbolic_funcs, numerical_funcs, constants):
    derivatives = {}
    numerical_derivatives = {}
    for match in re.findall(DERIVATIVE_PATTERN, expr):
        func_name, primes, wrt = match
        prime_count = len(primes)
        wrt_vars = list(wrt) if wrt else []
        if func_name not in symbolic_funcs:
            raise KeyError(f"Function '{func_name}' not found")
        if (n := len(wrt_vars)) and prime_count != n:
            raise SyntaxError("Number of variables must match the number of primes")
        func_expr = symbolic_funcs[func_name]
        variables = list(func_expr.free_symbols)
        try:
            if not wrt_vars and len(variables) == 1:
                var = variables[0]
                for _ in range(prime_count):
                    func_expr = sp.diff(func_expr, var)
            else:
                for var in wrt_vars:
                    func_expr = sp.diff(func_expr, sp.Symbol(var))
        except Exception as e:
            raise ValueError(f"Cannot differentiate function '{func_name}': {e}")
        new_func_name = f"{func_name}{'D' * prime_count}{''.join(wrt_vars)}"
        derivatives[new_func_name] = func_expr
        numerical_func = as_function(func_expr, numerical_funcs, constants)
        numerical_derivatives[new_func_name] = numerical_func
    expr = expr.replace("'", "D")
    return expr, derivatives, numerical_derivatives

def as_symbolic(arg, symbolic_funcs, numerical_funcs, constants):
    if isinstance(arg, sp.Basic):
        return arg
    if isinstance(arg, str):
        if "'" in arg:
            arg, derivatives, numerical_derivatives = rename_derivatives(
                arg, symbolic_funcs, numerical_funcs, constants
            )
            symbolic_funcs.update(derivatives)
            numerical_funcs.update(numerical_derivatives)
        arg = fix_expr(arg, symbolic_funcs, constants)
        local_dict = {}
        for name, func in symbolic_funcs.items():
            if isinstance(func, sp.Basic):
                # Wrap the expression as a Lambda function
                variables = sorted(func.free_symbols, key=lambda s: s.name)
                local_dict[name] = sp.Lambda(variables, func)
            else:
                local_dict[name] = func
        local_dict.update(constants)
        return sp.sympify(arg, locals=local_dict)
    if isinstance(arg, (numbers.Number, np.number)):
        return sp.sympify(arg)
    if callable(arg):
        sp_func = getattr(sp, arg.__name__, None)
        return sp_func if sp_func else sp.Function(arg.__name__)
    raise TypeError(f"Unsupported type for symbolic conversion: {type(arg)}")

def convert_functions(functions, constants):
    symbolic_funcs = {}
    numerical_funcs = {}
    for name in reversed(functions):
        obj = functions[name]
        if callable(obj):
            numerical_funcs[name] = obj
            sp_func = getattr(sp, name, None)
            symbolic_funcs[name] = sp_func if sp_func else sp.Function(name)
        else:
            sym_expr = as_symbolic(obj, symbolic_funcs, numerical_funcs, constants)
            variables = list(sym_expr.free_symbols)
            numerical_func = as_function(sym_expr, numerical_funcs, constants, variables)
            symbolic_funcs[name] = sym_expr
            numerical_funcs[name] = numerical_func
    return symbolic_funcs, numerical_funcs

def separate_definitions(definitions):
    functions = {}
    constants = {}
    for key in list(definitions):
        obj = definitions[key]
        if isinstance(obj, (numbers.Number, np.number)):
            constants[key] = obj
        else:
            functions[key] = obj
    return functions, constants

def as_function(expr, numerical_funcs, constants, variables=None):
    if callable(expr):
        return expr
    if not isinstance(expr, sp.Basic):
        raise TypeError(f"Expected sympy expression, got {type(expr)}")
    variables = variables or sorted(expr.free_symbols, key=lambda s: s.name)
    sym_expr = expr.subs(constants)
    return sp.lambdify(variables, sym_expr, modules=[numerical_funcs, 'numpy'])

class Function:
    global_symbolic_funcs = {}
    global_numerical_funcs = {}
    global_constants = {}

    def __init__(self, expression, **definitions):
        self.symbolic_funcs = {}
        self.numerical_funcs = {}
        functions, constants = separate_definitions(definitions)
        self.constants = {**self.global_constants, **constants}
        symbolic_funcs, numerical_funcs = convert_functions(functions, self.constants)
        self.symbolic_funcs.update(self.global_symbolic_funcs)
        self.symbolic_funcs.update(symbolic_funcs)
        self.numerical_funcs.update(self.global_numerical_funcs)
        self.numerical_funcs.update(numerical_funcs)
        self._symbolic_expression = as_symbolic(
            expression, self.symbolic_funcs, self.numerical_funcs, self.constants
        )
        self._function = as_function(
            self._symbolic_expression, self.numerical_funcs, self.constants
        )
    
    @classmethod
    def include(cls, module):
        if module=="numpy":
            Function.define(**{k:v for k,v in np.__dict__.items() if isinstance(v,np.ufunc) and v.nargs==2})
        else:
            raise ValueError("Module {module} not supported")

    @classmethod
    def define(cls, **definitions):
        functions, constants = separate_definitions(definitions)
        cls.global_constants.update(constants)
        symbolic_funcs, numerical_funcs = convert_functions(functions, cls.global_constants)
        cls.global_symbolic_funcs.update(symbolic_funcs)
        cls.global_numerical_funcs.update(numerical_funcs)

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)
