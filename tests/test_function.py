import numpy as np
import re
from symbolic_functor.function import fix_expr,Function
import sympy as sp
import symbolic_functor.utils as utils
import functools as fts
import pytest

@pytest.fixture
def constants():
    return dict(
        i = 1j,
        pi = np.pi
    )

@pytest.fixture
def functions():
    return dict(
        sin = np.sin,
        f = "x^1+2",
        g = lambda x,y:x**2+y**2,
    )

def test_fix_expr(functions, constants):
    test_cases = {
        "x^2y^2z":"x**2*y**2*z",
        "piit":"pi*i*t",
        "xsin(xyz)":"x*sin(x*y*z)",
        "ix+y(2+x)(x)(x-1)":"i*x+y*(2+x)*(x)*(x-1)",
        "3.1i+25.24sin(3x)":"3.1*i+25.24*sin(3*x)",
        "x4":"x4", #x*4 not supported
        "(x+1)2":"(x+1)2", #(x+1)*2 not supported
    }

    for input_arg, wanted_output in test_cases.items():
        assert fix_expr(input_arg, functions, constants) == wanted_output

def test_functions_with_derivatives_and_definitions(constants,functions):

    f = Function("x^2")
    assert f(3) == 9

    f = Function("f(x+1)",f = "x^2-1")
    assert f(3) == 15

    f = Function("f'(x-1)",f = "x^3")
    assert f(4) == 27

    f = Function("f''xy(w,z)",f = "x^2y^3")
    assert f(2,1) == 12

    f = Function("f'''(g'(x))",f = "x^4", g="x^2+x")
    assert f(2) == 120

    f = Function("f(x+1)",f="g(x+1)",g="x^2")
    assert f(1) == 9

def test_numpy():
    Function.define(**{k:v for k,v in np.__dict__.items() if isinstance(v,np.ufunc) and v.nargs==2})
    f  = Function("cos(x)tan(x)log(x)exp(x)")
    x = 0.4
    assert np.isclose(f(x),np.cos(x)*np.tan(x)*np.log(x)*np.exp(x))

# def test_tensor_input():
#     f = Function(["t^2", "t"])
#     assert np.all(np.isclose(f(2), np.array([4, 2])))

# def test_matrix_input():
#     Function.include("numpy")
#     f = Function([["cos(v)", "sin(v)"], ["-sin(v)", "cos(v)"]])
#     v = np.pi / 4
#     result = f(v)
#     expected = [[np.cos(v), np.sin(v)], [-np.sin(v), np.cos(v)]]
#     assert np.allclose(result, expected)

# def test_rotation_derivative():
#     Function.include("numpy")

#     Function.define(f = [["cos(v)", "sin(v)"], ["-sin(v)", "cos(v)"]])
#     f = Function("f'(v)")
#     v = np.pi / 4
#     result = f(v)
#     expected = [[-np.sin(v), np.cos(v)], [-np.cos(v), -np.sin(v)]]
#     assert np.allclose(result, expected)



# def test_mult_pattern():
#     exprs = {
#         "3i":"3*i",
#         "3pi*x":"3*pi*x",
#         "xy^2-iy":"x*y**2-i*y",
#         "3.4ix+2z":"3.4*i*x+2*z",
#         "xsi(xy)i":"x*s*i*(x*y)*i",
#         "4(x+1)(3-y)":"4*(x+1)*(3-y)",
#         "3(x)+-4(2-4)":"3*(x)+-4*(2-4)",
#         "iii":"i*i*i",
#         "xt^2y^2-y^2t":"x*t**2*y**2-y**2*t",
#         "(iy)":"(i*y)",
#         "isin(x)": "i*sin(x)",
#         "jtan(iy)": "j*tan(i*y)",
#         "2sin2(x)": "2*sin2(x)",
#         "2e^3": "2*e**3",
#         "2pi*i":"2*pi*i"
#     }

#     Function.define(pi = 3.14159, i=1j, e=2.718)
#     Function.define(sin2 = lambda x: np.sin(x)**2, sin = np.sin, tan = np.tan)
    
#     for original_expr, expected_expr in exprs.items():
#         new_expr = Function.fix_expr(original_expr.replace("^","**"), Function.global_constants, Function.global_functions)
#         assert new_expr == expected_expr, f"Expected {expected_expr}, but got {new_expr}"

#     Function.define(xyz = lambda x,y,z: x+y+z)
#     assert Function.fix_expr("xyz(xy,0,0)", Function.global_constants, Function.global_functions) == "xyz(x*y,0,0)"


# def test_with_definition():
#     f = Function("x^2*g(x, y)", g="x^2-y")
#     assert f(2, 3) == 4

# def test_with_constants():
#     f = Function("ax^2 + b*func_name(x^2)", a=4.5, b=2.25, func_name="x^2")
#     assert f(2) == 54

# def test_adding_global_functions_and_constants():
#     Function.define(f = "x^2", i=1j)
#     f = Function("f(x+1)f(y+1)f(z+1)i")
#     assert f(0,1,2) == 36j

# def test_variable_substitution_redundancy():
#     f = Function("f(u,u)", f="x^2 + y")
#     assert f(2) == 6, f"Expected f(2) to be 6, but got {f(2)}"

# def test_operations_with_self_numbers_and_strings_and_functions():

#     fa = Function("x^2") + 1
#     assert fa(2) == 5

#     fs = Function("x^2") - 1
#     assert fs(2) == 3

#     fm = Function("x^2") * 2
#     assert fm(2) == 8

#     fd = Function("x^2") / 2
#     assert fd(2) == 2

#     fp = Function("x^2") ^ 2
#     assert fp(2) == 16



#     fa = 1 + Function("x^2")
#     assert fa(2) == 5

#     fs = 1 - Function("x^2")
#     assert fs(2) == -3

#     fm = 2 * Function("x^2")
#     assert fm(2) == 8

#     fd = 2 / Function("x^2")
#     assert fd(2) == 1/2

#     fp = 2 ^ Function("x^2")
#     assert fp(2) == 16

# def test_derive_externally():
#     functions = {"f":lambda x:x**2,
#                  "g":"x^2y^3",
#                  "h":"x^2y^3z^4"}
#     expression = "f'(x) + g''yy(x, y) + h'''xyz(x, y, z)"
#     f = Function(expression, **functions)
#     assert f(1,2,3) == 2606

# def test_derivate():
    
#     assert sp.sympify("f(x)",locals = {"f":sp.Function("0")}).free_symbols == {sp.Symbol('x')}
    # Function.define(f = "x^2")
    # Function.define(g = "x^2y^3")
    # # f = Function("f'(x+1)")
    # # assert f(2) == 6

    # # f = Function("f'(x)")
    # # assert f(3) == 6 

    # f = Function("f''xy(x)")
    # assert f(3) == 0

    # f = Function("f'''(x)")
    # assert f(3) == 0