import unittest
from AMP import amp, funcEval
import math
import cmath
import decimal

numericalExpressions = {("2", 2), ("-2", -2), ("0", 0), ("2+3", 5), ("2-3", -1), ("2*3", 6), ("2/3", 2.0/3.0), ("2^3", 8), ("-2^6", -64), ("(-2)^6", 64), ("2%3", 2), ("3%2", 1), ("3!", 6), ("2!", 2),
    ("2.2", 2.2), ("-2.2", -2.2), ("2.3+4.5", 6.8), ("2.3-4.5", -2.2), ("-2.2*-3.32", 2.2*3.32), ("-2.1/-3.1", 2.1/3.1), ("2.1^2", 2.1**2), ("8.6%3.4", 1.8), (".5!", math.gamma(1.5)),
    ("i", complex("j")), ("-i", complex("-j")), ("-2+3i", complex("-2+3j")), ("(-2+3i)-(4-5i)", complex("-6+8j")), ("(-2+3i)+(4-5i)", complex("2-2j")), ("(-2+3i)*-(4-5i)", complex("-7-22j")),
    ("-(-2+3i)/(4-5i)", complex("2-3j")/complex("4-5j")), ("-(-2+3i)^-(4-5i)", -complex("-2+3j")**(-complex("4-5j"))), ("(-2+3i)%(4-5i)", complex("2-2j")), ("i!", funcEval.gamma(complex("1+j"))),
    ("(-2.3i-4)!", funcEval.gamma(complex("-3-2.3j"))), ("(2)", 2), ("-(2)", -2), ("-(-2)", 2), ("7(8)", 56), ("(7)(8)", 56), ("-(-7)(-8)", -56), ("-(-7)-(-8)", 15), ("2^-(3)", 0.125),
    ("2+3-4*5/6^7%8!", decimal.Decimal(349915.0)/decimal.Decimal(69984.0)), ("2.3e9+3e-2", 2300000000.03), ("-.02e-4-13.4e-15*(-(-.02e-4+13.4e-15)))", -0.000002), ("10 + 2 * 6", 22), ("100 * 2 + 12", 212),
    ("100 * ( 2 + 12 )", 1400), ("100 * ( 2 + 12 ) / 14", 100), ("((4))+3", 7), ("6+5*(3-2)", 11), ("10*(2+6)", 80), ("2*(-2)", -4), ("-1^(-3*4/-6)", -1), ("-2^(2^(4-1))", -256), ("2*6/4^2*4/3", 1),
    ("2^3^2", 512), ("1.1+2.2+10^2^3", 100000003.3),}

functionExpressions = {("sin(2)+cos(pi)", math.sin(2)-1), ("trunc(sin(2i)+cos(i*pi))", complex("11+3j")), ("exp(i*42)-cos(42)-isin(42)", 0),
    ("abs(fac(5i)+2)", decimal.Decimal(1.9983007972917787132074707595)), ("ceil(abs(fac(5i)+2))", 2), ("log(2+max(3, 3, 4))-sum(2, 1, 3)", decimal.Decimal(-5.221848749616356367))}

class TestStringMethods(unittest.TestCase):

    def test_numericalExpressions(self):
        for expr in numericalExpressions:
            self.assertEqual(amp.evaluate(expr[0]), amp.numberFormat(str(expr[1])), expr[0] + " != " + amp.evaluate(expr[0]) + " -> " + expr[0] + " = " + amp.numberFormat(str(expr[1])))
                    
    def test_functionExpressions(self):
        for expr in functionExpressions:
            self.assertEqual(amp.evaluate(expr[0]), amp.numberFormat(str(expr[1])), expr[0] + " != " + amp.evaluate(expr[0]) + " -> " + expr[0] + " = " + amp.numberFormat(str(expr[1])))

def runTests():
    unittest.main()

unittest.main()
