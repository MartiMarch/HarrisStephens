import cmath
import math
import random

precision = 12
_digits = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
openBrackets = {"(", "{", "["}
closeBrackets = {")", "}", "]"}
funcTwoArgs = {"mod", "rand", "randint"}
funcMultArgs = funcTwoArgs.union({"sum", "mean", "product", "min", "max", "range",
                        "median", "mode", "variance", "stdev", "gcd", "lcm", "pow"})
funcComplexArgs = {"sin", "cos", "tan", "asin", "acos", "atan", "csc", "sec", "cot", 
"ln", "log", "abs", "sqr", "exp", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
"ceil", "floor", "trunc", "sum", "mean", "product", "mode", "variance", "stdev", "gcd",
                                                        "lcm", "mod", "pow", "fac"}
functions = {"sin", "cos", "tan", "asin", "acos", "atan", "csc", "sec", "cot", 
"ln", "log", "abs", "sqr", "exp", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
"ceil", "floor", "trunc", "sum", "mean", "product", "min", "max", "range", "median",
"mode", "variance", "stdev", "gcd", "lcm", "mod", "pow", "rand", "randint", "fac"}

# Function Definitions

def gcd(array):
        if len(array) == 0:
                return 1
        elif len(array) == 1:
                return array[0]
        elif len(array) == 2:
                a = array[0]
                b = array[1]
                if isinstance(a, complex) or isinstance(b, complex):
                        return gcd([complex(str(a)).real, complex(str(a)).imag, complex(str(b)).real, complex(str(b)).imag])
                a = abs(a)
                b = abs(b)
                if a == 0 or b == 0:
                        return max(a, b)
                if (a % b == 0 or b % a == 0):
                        return min(a, b)
                newNum = max(a, b) - (long)(max(a, b) / min(a, b)) * min(a, b)
                return gcd([newNum, min(a, b)])
        else:
                return gcd([gcd(array[:len(array) / 2]), gcd(array[len(array) / 2:])])

def lcm(array):
        if len(array) == 0:
                return 0
        if len(array) == 1:
                return array[0]
        elif len(array) == 2:
                return array[0] * array[1] / gcd(array)
        else:
                return lcm([lcm(array[:len(array) / 2]), lcm(array[len(array) / 2:])])

# Coefficients used by the GNU Scientific Library
_g = 7
_p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
     771.32342877765313, -176.61502916214059, 12.507343278686905,
     -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]

def gamma(z):
        # Uses Lanczos Approximation to calculate Gamma Function
        z = complex(z)
        # Reflection formula
        if z.real < 0.5:
                return cmath.pi / (cmath.sin(cmath.pi * z) * gamma(1 - z))
        else:
                z -= 1
                x = _p[0]
                for i in range(1, _g + 2):
                        x += _p[i] / (z + i)
                t = z + _g + 0.5
        return cmath.sqrt(2 * cmath.pi) * (t ** (z + 0.5)) * cmath.exp(-t) * x

# Private Function Evaluation Helper Methods

def _replaceAltFuncSpellings(expression):
        expression = expression.replace("sine", "sin").replace("cosine", "cos").replace("tangent", "tan")
        expression = expression.replace("cosecant", "csc").replace("secant", "sec").replace("cotangent", "cot")
        expression = expression.replace("arc", "a").replace("sqrt", "sqr").replace("integer", "int")
        expression = expression.replace("flr", "floor").replace("ceiling", "ceil").replace("truncate", "trunc")
        expression = expression.replace("average", "avg").replace("minimum", "min").replace("maximum", "max")
        expression = expression.replace("standarddev", "stdev").replace("power", "pow").replace("random", "rand")
        expression = expression.replace("modulo", "mod").replace("avg", "mean").replace("factorial", "fact")
        expression = expression.replace("fact", "fac")
        return expression

# Public Function Evaluation Methods

def oneArgFuncEval(function, value):
        # Evaluates functions that take a complex number input
        if function == "sin":
                return cmath.sin(value)
        elif function == "cos":
                return cmath.cos(value)
        elif function == "tan":
                return cmath.tan(value)
        elif function == "asin":
                return cmath.asin(value)
        elif function == "acos":
                return cmath.acos(value)
        elif function == "atan":
                return cmath.atan(value)
        elif function == "csc":
                return 1.0 / cmath.sin(value)
        elif function == "sec":
                return 1.0 / cmath.cos(value)
        elif function == "cot":
                return 1.0 / cmath.tan(value)        
        elif function == "ln":
                return cmath.log(value)
        elif function == "sqr":
                return cmath.sqrt(value)
        elif function == "abs":
                return cmath.sqrt((value.real ** 2) + (value.imag ** 2))
        elif function == "exp":
                return cmath.exp(value)
        if function == "sinh":
                return cmath.sinh(value)
        elif function == "cosh":
                return cmath.cosh(value)
        elif function == "tanh":
                return cmath.tanh(value)
        elif function == "asinh":
                return cmath.asinh(value)
        elif function == "acosh":
                return cmath.acosh(value)
        elif function == "atanh":
                return cmath.atanh(value)
        elif function == "ceil":
                return math.ceil(value.real) + complex(0, 1) * math.ceil(value.imag)
        elif function == "floor":
                return math.floor(value.real) + complex(0, 1) * math.floor(value.imag)
        elif function == "trunc":
                return math.trunc(value.real) + complex(0, 1) * math.trunc(value.imag)
        elif function == "fac":
                if value.imag == 0 and value < 0 and value.real == int(value.real):
                        return "Error: The factorial function is not defined on the negative integers."
                return gamma(value + 1)
        elif function == "log":
                return cmath.log10(value)
                                
def multArgsFuncEval(function, arguments):
        # Evaluates functions with multiple parameters
        if function == "sum" or function == "mean":
                answer = 0
                for arg in arguments:
                        answer += arg
                if function == "sum":
                        return answer
                return answer / len(arguments)
        elif function == "product":
                answer = 1
                for arg in arguments:
                        answer *= arg
                return answer
        elif function == "min":
                return min(arguments)
        elif function == "max":
                return max(arguments)
        elif function == "range":
                return max(arguments) - min(arguments)
        elif function == "median":
                return sorted(arguments)[len(arguments) / 2]
        elif function == "mode":
                return max(set(arguments), key = arguments.count)
        elif function == "variance" or function == "stdev":
                sum = 0
                for arg in arguments:
                        sum += arg
                mean = sum / len(arguments)
                answer = 0
                for arg in arguments:
                        answer += (arg - mean) ** 2
                if function == "variance":
                        return answer / len(arguments)
                return cmath.sqrt(answer / len(arguments))
        elif function == "gcd":
                return gcd(arguments)
        elif function == "lcm":
                return lcm(arguments)
        elif function == "mod":
                firstNum = arguments[0]
                secondNum = arguments[1]
                return firstNum + (math.ceil((-firstNum / secondNum).real) + complex(0, 1) * math.ceil((-firstNum / secondNum).imag)) * secondNum
        elif function == "pow":
                answer = arguments[0]
                for i in xrange(1, len(arguments)):
                        answer **= arguments[i]
                return answer
        elif function == "log":
                if len(arguments) == 1:
                        return cmath.log10(arguements[0])
                elif len(arguments) == 2:
                        return cmath.log(arguments[0], arguements[1])
                else:
                        return "Error: 'log' has too many arguments."
        elif function == "rand":
                return random.uniform(arguments[0], arguments[1])
        elif function == "randint":
                return random.randint(arguments[0], arguments[1])
                
                        
def evalFunc(expression, function):
        for digit in _digits:
                expression = expression.replace(digit + function, digit + "*" + function)
        for bracket in closeBrackets:
                expression = expression.replace(bracket + function, bracket + "*" + function)
        expression = expression.replace("i" + function, "i" + "*" + function).replace("j" + function, "j" + "*" + function)
        
        # Finds the opening and closing brackets for the special mathematical function
        while expression.find(function) != -1:
                functionIndex = expression.find(function)
                bracketCount = 1
                openBracketIndex = functionIndex + len(function)
                closeBracketIndex = openBracketIndex + 1
                for char in expression[openBracketIndex + 1:]:
                        if char in openBrackets:
                                bracketCount += 1
                        elif char in closeBrackets:
                                bracketCount -= 1
                        if bracketCount == 0:
                                break
                        closeBracketIndex += 1
                answer = 0
                
                # Evaluates subexpressions
                for bracket in openBrackets:
                        if expression[openBracketIndex + 1 : closeBracketIndex].find(bracket) != -1:
                                beginning = expression[:openBracketIndex + 1]
                                end = expression[closeBracketIndex:]
                                subExpr = evaluateFunctions(expression[openBracketIndex + 1 : closeBracketIndex])
                                expression = beginning + subExpr + end
                                closeBracketIndex = len(beginning) + len(subExpr)

                if function in funcMultArgs:
                        # Separates the parameteres of any function with multiple arguments into a list and evaluates function
                        args = [x.strip() for x in expression[openBracketIndex + 1 : closeBracketIndex].split(',')]
                        if len(args) == 0 or (len(args) < 2 and function in funcTwoArgs):
                                return "Error: '" + function + "' is missing arguments."
                        elif function not in funcMultArgs and len(args) != 1:
                                return "Error: '" + function + "' only takes one argument."
                        elif len(args) > 2 and function in funcTwoArgs:
                                return "Error: '" + function + "' has too many arguments."
                        for i in xrange(0, len(args)):
                                args[i] = complex(amp.evaluate(str(args[i])).replace("i", "j"))
                                if function not in funcComplexArgs:
                                        if complex(args[i]).imag != 0:
                                                return "Error: '" + function + "' does not except nonreal arguments."
                                        args[i] = args[i].real
                        answer = multArgsFuncEval(function, args)
                else:
                        # Evaluates the expression inside the brackets
                        value = complex(amp.eval(expression[openBracketIndex + 1 : closeBracketIndex]).replace("i", "j"))
                        if function not in funcComplexArgs:
                                if value.imag != 0:
                                        return "Error: '" + function + "' does not except nonreal arguments."                                       
                                value = value.real
                        answer = oneArgFuncEval(function, value)

                expression = expression[:functionIndex] + amp.complexFormat(answer) + expression[closeBracketIndex + 1:]
        return expression

def evaluateFunctions(expression):
        expression = _replaceAltFuncSpellings(expression)
        for func in functions:
                expression = evalFunc(expression, func)
        return expression

import amp
amp.precision = precision + 5
