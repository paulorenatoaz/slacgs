import os
from cryptography.fernet import Fernet


def get_grandparent_folder_path():
  current_script_path = os.path.abspath(__file__)  # Get the absolute path of the current script
  parent_folder_path = os.path.dirname(current_script_path)  # Get the parent folder path
  grandparent_folder_path = os.path.dirname(parent_folder_path)  # Get the grandparent folder path
  return grandparent_folder_path

def get_parent_folder_path():
  current_script_path = os.path.abspath(__file__)  # Get the absolute path of the current script
  parent_folder_path = os.path.dirname(current_script_path)  # Get the parent folder path
  return parent_folder_path

def read_text_from_file(file_path):
  try:
    with open(file_path, 'r') as file:
      content = file.read()
      return content
  except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    return None
  except IOError:
    print(f"Error: Unable to read the file '{file_path}'.")
    return None


def encrypt_content(key, content):
    fernet = Fernet(key)
    encrypted_content = fernet.encrypt(content.encode())
    return encrypted_content


def decrypt_content(key, encrypted_content):
  fernet = Fernet(key)
  decrypted_content = fernet.decrypt(encrypted_content).decode()
  return decrypted_content


def write_decrypted_content_to_key_file():
  encryption_key = 'OWclcgqnaTKbX1CYLQDNQIJ7M0IieEhJknF5X4KrmZ4='
  encrypted_content_path = get_parent_folder_path() + '\\enc.py'

  with open(encrypted_content_path, 'rb') as file:
    encrypted_content = file.read()

  decrypted_content = decrypt_content(encryption_key, encrypted_content)

  key_path =  get_grandparent_folder_path() + '\\slacgs\\key.py'
  with open(key_path, 'w') as key_file:
    key_file.write(decrypted_content)



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


def is_valid_email(email):
  # Regular expression pattern for email validation
  pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
  return re.match(pattern, email)