import numpy as np
import sympy as sp
from typing import Iterable, Callable
import re
import inspect
import symbolic_functor.utils as utils
import numbers


# def is_constant(value):
#     if isinstance(value, (int, float, complex)):
#         return True
#     return False

# class Function:
#     global_constants = {}
#     global_functions = {}
#     function_start = 0xE000
#     constant_start = 0xE100
#     implicit_mult_pattern = fr"(?<=[\da-zA-Z)\u{constant_start:X}-\uE1FF])([(a-zA-Z\u{function_start:X}-\uEFFF])"
#     derivative_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*?)(D+)([a-zA-Z]*)?\("

#     def __init__(self, arg, **definitions):
#         self.free_variables = set()
#         self.local_constants = {}
#         self.local_functions = {}
#         temp_variables = set()
#         self._process_definitions(definitions, temp_variables, self.local_constants, self.local_functions)
#         self.symbolic_expression = self.as_symbolic_expression(arg, self.free_variables, self.local_constants, self.local_functions)
#         self.function = self.as_function(self.symbolic_expression, self.local_constants, self.local_functions)


#     @classmethod
#     def define(cls,**definitions):
#         variables = set()
#         cls._process_definitions(definitions, variables, cls.global_constants, cls.global_functions)

#     @classmethod
#     def _process_definitions(cls, definitions, variables, constants, functions):
#         symbolic_expressions = {}

#         for name, value in definitions.items():
#             if callable(value):

#                 functions[name] = value
#             elif isinstance(value, (str, bytes)):
#                 sym_expr = cls.as_symbolic_expression(value, variables, constants, functions)
#                 if sym_expr.is_constant():
#                     constants[name] = sym_expr.evalf()
#                 else:
#                     symbolic_expressions[name] = sym_expr
#             elif is_constant(value):
#                 constants[name] = value
#             else:
#                 raise ValueError(f"Unsupported type for definition '{name}': {type(value)}")

#         functions.update(
#             {name: cls.as_function(expr,constants,functions) for name, expr in symbolic_expressions.items()}
#         )

#     @classmethod
#     def as_symbolic_expression(cls, value, variables, constants, functions):
#         if isinstance(value, bytes):
#             value = value.decode('utf-8')
#         if isinstance(value, str):
#             value = cls.fix_expr(value, constants, functions)
#         try:
#             locals = {**cls.all_constants(constants),**cls.all_functions(functions)}
#             sym_expr = sp.sympify(value, locals = locals, convert_xor=True)
#             variables.update(sym_expr.free_symbols)
#             return sym_expr
#         except sp.SympifyError:
#             raise ValueError(f"Cannot interpret the string '{value}' as a symbolic expression")
        
#     @classmethod
#     def all_constants(cls, constants):
#         return {**constants, **cls.global_constants}
    
#     @classmethod
#     def all_functions(cls, functions):
#         return {**functions, **cls.global_functions}

#     @classmethod
#     def as_function(cls, sym_expr, constants, functions, variables = None):
#         all_constants = cls.all_constants(constants)
#         all_functions = cls.all_functions(functions)
#         new_expr = sym_expr.subs(all_constants)
#         sorted_variables = sorted(new_expr.free_symbols if variables is None else variables,key=str)
#         return sp.lambdify(sorted_variables, new_expr, modules = [all_functions])

#     @classmethod
#     def fix_expr(cls, expr, constants, functions):
#         all_constants = cls.all_constants(constants)
#         all_functions = cls.all_functions(functions)

#         new_expr = expr.replace("'", "D")
#         cls.add_derivatives(new_expr, all_functions, functions)

#         all_functions.update(functions)

#         ordered_functions = sorted(all_functions, key=len, reverse=True)
#         ordered_constants = sorted(all_constants, key=len, reverse=True)

#         for index, named_func in enumerate(ordered_functions):
#             placeholder = chr(cls.function_start + index)
#             new_expr = new_expr.replace(named_func, placeholder)

#         for index, named_const in enumerate(ordered_constants):
#             placeholder = chr(cls.constant_start + index)
#             new_expr = new_expr.replace(named_const, placeholder)

#         new_expr = re.sub(cls.implicit_mult_pattern, r"*\1", new_expr)

#         for index, named_const in enumerate(ordered_constants):
#             placeholder = chr(cls.constant_start + index)
#             new_expr = new_expr.replace(placeholder, named_const)

#         for index, named_func in enumerate(ordered_functions):
#             placeholder = chr(cls.function_start + index)
#             new_expr = new_expr.replace(placeholder, named_func)

#         return new_expr
    
#     @classmethod
#     def add_derivatives(cls, expr, all_functions, functions):

#         for match in re.findall(cls.derivative_pattern, expr):
#             function_name = match[0] 
#             prime_count = len(match[1]) 
#             resp_vars = match[2] if match[2] else []
        
#             new_func = all_functions[function_name]

#             if isinstance(new_func,bytes):
#                 new_func = new_func.decode("utf-8")
#             if isinstance(new_func,str):
#                 new_func = cls.fix_expr(new_func,cls.global_constants,functions)
#                 locals = {**cls.global_constants,**functions}
#                 sym_expr = sp.sympify(new_func,locals=locals,convert_xor=True)
#                 available_vars = sym_expr.free_symbols
#             elif isinstance(new_func,Function):
#                 available_vars = new_func.free_variables
#                 sym_expr = new_func.symbolic_expression
#             elif callable(new_func):
#                 sig = inspect.signature(new_func)
#                 param_names = list(sig.parameters.keys())
#                 symbols = {name:sp.Symbol(name) for name in param_names}
#                 available_vars = set(symbols.values())
#                 sym_expr = new_func(**symbols)
#             else:
#                 raise ValueError("Not valid function")
            
#             variables = sym_expr.free_symbols
#             if not resp_vars and len(available_vars)==1:
#                 for _ in range(prime_count):
#                     sym_expr = sp.diff(sym_expr,list(available_vars)[0])
#             else:
#                 for var in resp_vars:
#                     symbol = sp.Symbol(var)
#                     sym_expr = sp.diff(sym_expr,symbol)
#             functions[function_name+"D"*prime_count+"".join(resp_vars)] = cls.as_function(sym_expr,{},functions,variables = variables)

#     def __repr__(self):
#         repr_expr = str(self.symbolic_expression)
#         repr_expr = repr_expr.replace("D", "'").replace("**",'^')
#         return repr_expr

#     def __call__(self, *args, **kwargs):
#         return self.function(*args,**kwargs)
    
#     def __add__(self, arg):
#         new_symbolic_expr = self.symbolic_expression + arg
#         return Function(new_symbolic_expr)

#     def __sub__(self, arg):
#         new_symbolic_expr = self.symbolic_expression - arg
#         return Function(new_symbolic_expr)
    
#     def __mul__(self, arg):
#         new_symbolic_expr = self.symbolic_expression * arg
#         return Function(new_symbolic_expr)
    
#     def __truediv__(self, arg):
#         new_symbolic_expr = self.symbolic_expression / arg
#         return Function(new_symbolic_expr)
    
#     def __xor__(self, arg):
#         new_symbolic_expr = self.symbolic_expression ** arg
#         return Function(new_symbolic_expr)
    
#     def __radd__(self, arg):
#         new_symbolic_expr = arg + self.symbolic_expression
#         return Function(new_symbolic_expr)

#     def __rsub__(self, arg):
#         new_symbolic_expr = arg - self.symbolic_expression
#         return Function(new_symbolic_expr)
    
#     def __rmul__(self, arg):
#         new_symbolic_expr = arg * self.symbolic_expression
#         return Function(new_symbolic_expr)
    
#     def __rtruediv__(self, arg):
#         new_symbolic_expr = arg / self.symbolic_expression
#         return Function(new_symbolic_expr)
    
#     def __rxor__(self, arg):
#         new_symbolic_expr = arg ** self.symbolic_expression
#         return Function(new_symbolic_expr)



FUNCTION_START = 0xE000
CONSTANT_START = 0xE100
IMPL_MULT_PATTERN = re.compile(fr"(?<=[\da-zA-Z)\u{CONSTANT_START:X}-\uE1FF])([(a-zA-Z\u{FUNCTION_START:X}-\uEFFF])")
DERIVATIVE_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*?)(\'+)([a-zA-Z]*)?\(")

def fix_expr(expr, functions, constants):
    ordered_functions = sorted(functions, key=len, reverse=True)
    ordered_constants = sorted(constants, key=len, reverse=True)

    new_expr = expr.replace("^","**")

    for index, named_func in enumerate(ordered_functions):
        placeholder = chr(FUNCTION_START + index)
        new_expr = new_expr.replace(named_func, placeholder)

    for index, named_const in enumerate(ordered_constants):
        placeholder = chr(CONSTANT_START + index)
        new_expr = new_expr.replace(named_const, placeholder)

    new_expr = IMPL_MULT_PATTERN.sub(r"*\1", new_expr)

    for index, named_const in enumerate(ordered_constants):
        placeholder = chr(CONSTANT_START + index)
        new_expr = new_expr.replace(placeholder, named_const)

    for index, named_func in enumerate(ordered_functions):
        placeholder = chr(FUNCTION_START + index)
        new_expr = new_expr.replace(placeholder, named_func)

    return new_expr

def rename_derivatives(expr:str, functions:dict, constants:dict) -> tuple[str,dict]:
    derivatives = {}
    for match in re.findall(DERIVATIVE_PATTERN, expr):
        func_name = match[0] 
        prime_count = len(match[1]) 
        with_respect_to = match[2] if match[2] else []
        if not func_name in functions:
            raise KeyError(f"function {func_name} not found")
        if (N:=len(with_respect_to)) and prime_count!=N:
            raise SyntaxError(f"Number of with-respect-to variables need to be 0 or match the number of primes")
        
        func = functions[func_name]
        sym_expr = as_symbolic(func,functions)
        variables = sym_expr.free_symbols
        try:
            if not with_respect_to and len(variables)==1:
                for _ in range(prime_count):
                    sym_expr = sp.diff(sym_expr,list(variables)[0])
            else:
                for var in with_respect_to:
                    sym_expr = sp.diff(sym_expr,sp.Symbol(var))
        except Exception:

            raise ValueError(f"The function {func_name} can't be differentiated symbolically")
            
        new_func_name = f"{func_name}{'D'*prime_count}{''.join(with_respect_to)}"
        derivatives[new_func_name] = as_function(sym_expr, functions, constants, variables)

    return expr.replace("'","D"), derivatives

def as_symbolic(arg, functions, constants={}):
    if isinstance(arg, sp.Basic):
        return arg
    if isinstance(arg, bytes):
        arg = arg.decode("utf-8")
    if isinstance(arg, str):
        if "'" in arg:
            arg, derivatives = rename_derivatives(arg, functions, constants)
            functions.update(derivatives)
        arg = fix_expr(arg, functions, constants)
        # Map function names to sympy functions where possible
        local_dict = {}
        for name, func in functions.items():
            if isinstance(func, (sp.FunctionClass, sp.Function)):
                local_dict[name] = func
            elif callable(func):
                # Try to map to sympy function
                sp_func = getattr(sp, name, None)
                if sp_func:
                    local_dict[name] = sp_func
                else:
                    local_dict[name] = sp.Function(name)
            else:
                local_dict[name] = sp.Function(name)
        local_dict.update(constants)
        return sp.sympify(arg, locals=local_dict)
    elif isinstance(arg, (numbers.Number, np.number)):
        return sp.sympify(arg)
    elif callable(arg):
        # Try to map to sympy function
        sp_func = getattr(sp, arg.__name__, None)
        if sp_func:
            return sp_func
        else:
            return sp.Function(arg.__name__)
    else:
        sig = inspect.signature(arg)
        param_names = list(sig.parameters.keys())
        symbols = {name: sp.Symbol(name) for name in param_names}
        return arg(**symbols)


    
def convert_functions(functions, constants):
    new_functions = {}
    function_names = list(functions.keys())
    # Process functions from right to left
    for name in reversed(function_names):
        obj = functions[name]
        # Collect already processed functions
        known_functions = {**new_functions}
        try:
            if isinstance(obj, sp.FunctionClass):
                # If it's already a SymPy function, store it directly
                new_functions[name] = obj
            elif callable(obj):
                # Store callable functions (like NumPy functions) as-is
                new_functions[name] = obj
            else:
                # Convert the function expression to symbolic form
                sym_expr = as_symbolic(obj, known_functions, constants)
                variables = list(sym_expr.free_symbols)
                # Create a callable function using lambdify
                new_functions[name] = as_function(sym_expr, known_functions, constants, variables)
        except Exception as e:
            raise ValueError(f"Error converting function '{name}': {e}")
    return new_functions



def separate_definitions(definitions):
    functions = {}
    constants = {}
    keys = list(definitions.keys())
    for key in keys:
        obj = definitions.pop(key)
        if isinstance(obj,(numbers.Number,np.number)):
            constants[key] = obj
        else:
            functions[key] = obj
    return functions, constants

    
def as_function(arg, functions, constants, variables=None):
    if callable(arg):
        # Return the callable function as-is
        return arg
    if isinstance(arg, sp.FunctionClass):
        # Return the SymPy function class as-is
        return arg
    if not isinstance(arg, sp.Basic):
        arg = as_symbolic(arg, functions, constants)
    if variables is None:
        variables = arg.free_symbols
    sorted_variables = sorted(variables, key=str)
    sym_expr = arg.subs(constants)
    utils.log("before lambdify", sym_expr)
    return sp.lambdify(sorted_variables, sym_expr, modules=[functions])


class Function:
    global_functions = {}
    global_constants = {}

    def __init__(self, expression, **definitions):
        all_constants, converted_functions = self._prepare_definitions(definitions)

        self._symbolic_expression = as_symbolic(expression, converted_functions, all_constants)
        utils.log(self._symbolic_expression, converted_functions, all_constants)
        self._function = as_function(self._symbolic_expression, converted_functions, all_constants)

    @classmethod
    def define(cls, **definitions):
        all_constants, converted_functions = cls._prepare_definitions(definitions)
        cls.global_functions.update(converted_functions)
        cls.global_constants.update(all_constants)
    
    @classmethod
    def include(cls, module):
        if module=="numpy":
            Function.define(**{k:v for k,v in np.__dict__.items() if isinstance(v,np.ufunc) and v.nargs==2})
        else:
            raise ValueError("Module {module} not supported")

    @classmethod
    def _prepare_definitions(cls, definitions):
        functions, constants = separate_definitions(definitions)
        all_functions = {**cls.global_functions, **functions}
        all_constants = {**cls.global_constants, **constants}
        converted_functions = convert_functions(all_functions, all_constants)
        return all_constants, converted_functions


    def __call__(self,*args,**kwargs):
        return self._function(*args,**kwargs)
                



            
        



