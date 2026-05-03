# from langchain.tools import tool
# class CalculatorTools():
#     @tool("Make a calculation")
#     def calculate(operation):
#         """This tool allows you to perform mathematical operations like 
#             addition, subtraction, multiplication, and division. 
#             It takes a mathematical expression as input, such as 150+25 or 300/5*2."""
#         try:
#             return eval(operation)
#         except SyntaxError:
#             return "Error: Invalid syntax in mathematical expression"
        
#As per our version of crew-ai
from crewai.tools import tool

class CalculatorTools:

    @tool("Make a calculation")
    def calculate(operation: str) -> str:
        """
        Perform mathematical calculations.
        Example: 150+25, 300/5*2
        """
        try:
            return str(eval(operation))
        except Exception:
            return "Error: Invalid mathematical expression"
