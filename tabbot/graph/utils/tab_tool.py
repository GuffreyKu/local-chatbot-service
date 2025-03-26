import textwrap
from langchain_core.tools import tool


@tool(parse_docstring=True)
def code_model(command:str):
    """Execute a Python command.

    This function attempts to evaluate the given command using eval. If eval raises a SyntaxError,
    the command is then executed using exec. If the command returns a value, it prints the result.

    Args:
        command (str): The Python command to execute.

    Returns:
        Any: The result of the evaluated command, if any.
    """
    try:
        # Try to compile and evaluate as an expression
        compiled_code = compile(command, "<string>", "eval")
        result = eval(compiled_code)
    except SyntaxError:
        # If it's not a valid expression, compile and execute as a statement
        compiled_code = compile(command, "<string>", "exec")
        local_vars = {}
        exec(compiled_code, globals(), local_vars)
        # Optionally retrieve a 'result' variable if it's defined in the executed code
        result = local_vars.get("result", None)

    return result

if __name__ == '__main__':
    import textwrap

    command_loop = """\
import numpy as np
result = np.zeros(5)
    """

    result = code_model.invoke(command_loop)
    print("Returned:", result)      