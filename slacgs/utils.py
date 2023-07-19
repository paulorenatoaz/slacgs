import os

def is_jupyter_notebook():
  """Check if the environment is a Jupyter notebook."""
  try:
    # Check if 'get_ipython' function exists
    from IPython import get_ipython
    # Check if running in a notebook
    if get_ipython() is not None:
      if 'IPKernelApp' in get_ipython().config:
        return True
  except ImportError:
    pass
  return False


def is_colab_notebook():
  """Check if the environment is a Google Colab notebook."""
  try:
    import google.colab
    return True
  except ImportError:
    return False


def is_notebook():
  """Check if the environment is a notebook."""
  return is_jupyter_notebook() or is_colab_notebook()

def cls():
  """
  Clears the terminal screen. Works on both Windows and Linux.
  """
  if os.name == 'nt':  # For Windows
    _ = os.system('cls')
  else:  # For Linux and Mac
    _ = os.system('clear')

