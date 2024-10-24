import json
import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.fernet import Fernet
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pygsheets


## this is a report service configuration dictionary

def get_user_folder_path():
  """Returns the user's home folder in any operating system.

  Returns:
    str: The user's home folder.

  """
  return os.environ['USERPROFILE'] if os.name == 'nt' else os.environ['HOME']


def is_jupyter_notebook():
  """Check if the environment is a Jupyter notebook.

  Returns:
    bool: True if the environment is a Jupyter notebook, False otherwise.

  """
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
  """Check if the environment is a Google Colab notebook.

  Returns:
    bool: True if the environment is a Google Colab notebook, False otherwise.

  """
  try:
    import google.colab
    return True
  except ImportError:
    return False


def is_notebook():
  """Check if the environment is a notebook.

  Returns:
    bool: True if the environment is a notebook, False otherwise.

  """
  return is_jupyter_notebook() or is_colab_notebook()


"""
report_service_conf: dict
    A dictionary holding the configuration for the report service. It contains the following keys:

    - 'images_path': str, the path where images are stored. This path depends on whether the code is running on Google Colab, Windows, or another operating system.
    - 'reports_path': str, the path where reports are stored. This path also depends on the operating environment.
    - 'user_email': str or None, the user's email used for Google services authentication. It defaults to None.
    - 'drive_service': object or None, the Google Drive service object. It defaults to None.
    - 'spreadsheet_service': object or None, the Google Sheets service object. It defaults to None.
    - 'pygsheets_service': object or None, the Pygsheets service object. It defaults to None.

    This dictionary is used throughout the module to configure and use Google services.
"""

report_service_conf = {
  'output_path' : '/content/slacgs/output' if is_colab_notebook()
  else os.path.join(os.path.expanduser("~"), 'slacgs', 'output'),
  'images_path': '/content/slacgs/images/' if is_colab_notebook()
  else os.path.join(os.path.expanduser("~"), 'slacgs', 'output', 'images'),
  # else os.path.join(os.path.dirname(__file__), '..', 'output','reports', 'images'),
  'visualizations_path': '/content/slacgs/output/report/images/visualizations' if is_colab_notebook()
  else os.path.join(os.path.expanduser("~"), 'slacgs', 'output', 'reports', 'images', 'visualizations'),
  # else os.path.join(os.path.dirname(__file__), '..', 'output', 'reports', 'images', 'visualizations'),
  'graphs_path': '/content/slacgs/output/reports/images/graphs' if is_colab_notebook()
  else os.path.join(os.path.expanduser("~"), 'slacgs', 'output', 'reports', 'images', 'graphs'),
  # else os.path.join(os.path.dirname(__file__), '..', 'output', 'reports', 'images', 'graphs'),
  'tables_path': '/content/slacgs/output/reports/tables' if is_colab_notebook()
  else os.path.join(os.path.expanduser("~"), 'slacgs', 'output', 'reports', 'tables'),
  # else os.path.join(os.path.dirname(__file__), '..', 'output', 'reports', 'tables'),
  'reports_path': '/content/slacgs/reports/' if is_colab_notebook()
  else os.path.join(os.path.expanduser("~"), 'slacgs', 'output', 'reports'),
  # else os.path.join(os.path.dirname(__file__), '..', 'output', 'reports'),
  'user_email': None,
  'drive_service': None,
  'spreadsheet_service': None,
  'pygsheets_service': None
}


def is_param_in_simulation_reports(params):
  """Check if a parameter is already in the simulation_reports.json file.

  Parameters:
      params (list): The parameter to be checked.

  Returns:
      bool: True if the parameter is already in the simulation_reports.json file, False otherwise.
  """
  if not os.path.exists(os.path.join(report_service_conf['output_path'], 'simulation_reports.json')):
    return False

  with open(os.path.join(report_service_conf['output_path'], 'simulation_reports.json'), 'r') as f:
    simulation_reports = json.load(f)

  for report in simulation_reports:
    if report['model_tag']['params'] == params:
      return True

  return False


def set_report_service_conf(path_to_google_cloud_service_account_api_key=None, user_google_account_email=None,
                            slacgs_password=None):
  """Set the report service configuration. This function must be called before using the report service dependencies models (e.g. GspreadClient, GdriveClient).

  Parameters:
    slacgs_password (str): The password used to enable our Report Service. Defaults to None.
    user_google_account_email (str): The user email used to authenticate the Google services. Defaults to None.
    path_to_google_cloud_service_account_api_key (str): The Google Cloud Service Account API Json Key used to build your own Report Service. Must be able to access Google Drive and Google Sheets API's. Defaults to None.

  Returns:
    None
    
  Raises:
    TypeError: If slags_password is not a string.
    TypeError: If user_google_account_email is not a string.
    TypeError: If google_cloud_service_account_api_key is not a string.

  Observations:
    If slags_password or user_google_account_email is None, then the user will be prompted to enter the password or email.
    If google_cloud_service_account_api_key is given, then the password will be ignored.

  """

  if slacgs_password is not None and not isinstance(slacgs_password, str):
    raise TypeError("slags_password must be a string.")

  if user_google_account_email is not None and not isinstance(user_google_account_email, str):
    raise TypeError("user_google_account_email must be a string.")

  if path_to_google_cloud_service_account_api_key is not None and not isinstance(path_to_google_cloud_service_account_api_key, str):
    raise TypeError("google_cloud_service_account_api_key must be a string.")

  if user_google_account_email is not None:
    report_service_conf['user_email'] = user_google_account_email

  SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']

  if path_to_google_cloud_service_account_api_key is not None:
    credentials = service_account.Credentials.from_service_account_file(path_to_google_cloud_service_account_api_key, scopes=SCOPES)
  else:
    key_obj = eval(get_key(slacgs_password))
    credentials = service_account.Credentials.from_service_account_info(key_obj, scopes=SCOPES)

  report_service_conf['pygsheets_service'] = pygsheets.authorize(custom_credentials=credentials)
  report_service_conf['drive_service'] = build('drive', 'v3', credentials=credentials)
  report_service_conf['spreadsheet_service'] = build('sheets', 'v4', credentials=credentials)


def get_grandparent_folder_path():
  """Get the grandparent folder path of the current script.

  Returns:
      str: The grandparent folder path of the current script.
  """

  current_script_path = os.path.abspath(__file__)  # Get the absolute path of the current script
  parent_folder_path = os.path.dirname(current_script_path)  # Get the parent folder path
  grandparent_folder_path = os.path.dirname(parent_folder_path)  # Get the grandparent folder path
  return grandparent_folder_path


def get_parent_folder_path():
  """Get the parent folder path of the current script.

  Returns:
      str: The parent folder path of the current script.
  """

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


def derive_key(password, salt):
  kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=salt,
    iterations=100000,
    backend=default_backend()
  )
  key = kdf.derive(password.encode('utf-8'))
  return key


def encrypt_key(data, password):
  salt = os.urandom(16)
  key = derive_key(password, salt)

  padder = padding.PKCS7(algorithms.AES.block_size).padder()
  padded_data = padder.update(data.encode('utf-8')) + padder.finalize()

  iv = os.urandom(16)
  cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
  encryptor = cipher.encryptor()
  ciphertext = encryptor.update(padded_data) + encryptor.finalize()

  return salt + iv + ciphertext

def encrypt_content(content, password):
  key = Fernet.generate_key()
  fernet = Fernet(key)
  encrypted_content = fernet.encrypt(content.encode())
  return encrypt_key(key, password), encrypted_content

def decrypt_key(password):
  encrypted_data = b'\xfb\xaf\xadX\r\xe1c\xbe\xdf\x1f\xb8d?j\x1fF\x94\xbf\x1f\xf4"\x12~\r\x19\xc6\x9ah\xa2\x08\xf8\xadO\xd8\xdb\x969x\x954\x05\xa1vn\xe6a`\x94\xa9l\x99uo:\xd3\r\xc7\x08\x1d\x85\xd4\xef\x8d\xcdg\x04iJi\xa1\x04<\x86\xf2\xcea\xa5R\xd9\x10'
  salt = encrypted_data[:16]
  iv = encrypted_data[16:32]
  ciphertext = encrypted_data[32:]

  key = derive_key(password, salt)

  cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
  decryptor = cipher.decryptor()
  padded_data = decryptor.update(ciphertext) + decryptor.finalize()

  unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
  data = unpadder.update(padded_data) + unpadder.finalize()

  return data.decode('utf-8')


def recover(key, content):
  fernet = Fernet(key)
  recovered = fernet.decrypt(content).decode()
  return recovered



def get_key(password=None):
  """Get the key.
  Args:
    password (str): The password to decrypt the key.

  Returns:
    str: The key content.

  """

  content = b'gAAAAABkuz39ZGqF4hbNnBAIkDI50gNTHw5GRFMvQR7E4eihvZQZZBoR53OviT96w46c6Zd7NekTq9dpN9886wfb4qNmmswrnqi_ar8C0TL5XPjUbzSwunTPhvE203c9YP-8WVo3wPY6cpakOeXCkShMd5W6VPb4-An4v49qmc02B-CFXIzoyraO6gWVh7UQ_LhPszOKMLuyOXkUf3O1KCrJ_cbJSLRNX2Gn2JAGRH6lYe9qY-8QRmZQ3JpuDNsvcm7Df47pdW4VZ86fop1Ut5ACdfiRE8O_47YmLQTFj1bnhcU1BW_KDT9eP7qDVhsRiRGJwS_12PIyN_R6IQqX_Q7VXpd4DZw2m9rlGa3fgfGT6JHaBU7S0zf7neR8rGFTJm0OsqPt--1cB_5ptIHXCKh_xs7FA3jsubdv94vpIMM6a9ocWpd9uWxkxVK38cYYc15QbIuBfrsegAhbwewTLFRc4r6wOXW1hGnMqPgccwrYQEVMpIQmSXiABr6bifIFBtYcQ3se4bVllPF703y4VKsQMgsS-uCPZHBPc5lHO2mKbHxHmEE3jG1rO4HbRM1BLR0luGW48nLycKxkZV7Tp0VnnLBYqiTK9cZbNzSeCqb1Awd5Tz0xiFLVjLmXOeRPUc89yG8gXmdU9Apl0A9UIeopZ9r5GIpKdv_w70R9yrOeItATttMie2_81hy8xG4nLBfmbetUaXf-b0ab6PJA50wlZLwVhEV3xveLd9bhwyNLlWK4XFQkOW9AIohsyPZq7niy3KEvimtuoDAU6dtAkV5hItTplCLfvNATTq7cSByVeZSAB0beHO2vMDINBoCqqmwP5OPnNrgJc0J4nAM5rDYlIXhpnZMTLqVYN4FNReBm-dz7ykPBFmdry8nugA2ConfnrwSYm5TjwMr8yyMrdnAk0UARxjNNV-fN9Zb3y-RShWgsRR1Hz1bfQdV76MpE-_D4HKmLlEpLaVuBJsWa5w0kW9HJ9tkSpoE56T6G9iRBZKa8gcF3nPo2AHGRnAAKwh3dMdac1nF-1yt-WeBpYl1s8UnJNtQX8oEnrNiU69WuFtMd9XSV-kQ972cuZnIOVMmQcWu8U8LnSljUnQ42kXPsu1kqeqRnyTAuGqId2YWSBc8fSx54Ru0fs-FpIV_vEugTyeIe2iRbP5H79RwgQIo9sdZ8VQ-sq3aezqdM3A7jT9YQ6LV9cShQTkXBUxi0eeH4pTJm8lWO96IJ1PaBnQ966-9VDwxw0Ns-Sc-KSBNpjLgx3hOeoRUN9EDqUBaTmzJJFkpdHDHAml_cMh4CNRkC_tpL_BU7e7eqih8_uPj9ilxsM__06Bfwj45p3kQzmx2LJTAQwESPfygAqntMuK_ODz_ZFjuPlJfeg8ttXwqlgQV4QogxeOtcWTSVqcWQwAxVgNZi7s-VLOnjzPPi2PlnXrK7zUDNemgakWaUCA0Pcekwga8mBLIkLpts19tXohnoPzQo0eluJskwQ5Jg3p0Tzm5CnLeEHnI59WOpkSWeZeOllCmxNxxo4tcb2eiKy4gywOyXZvKJj5AzRMEfWDnPrb_zZ5narqiu2fTFJC7i4BeIybLXLiCry7UZSGEKaSq690lEYonFillHnuPB7gexNVb8xgpAeujfeZ_aMkousKXuXtSIcXrmLQi1cdYEH8H3_mEXUSRC09rBFkLBu2qZqR--TUQiKo1Zd_fyjEknvgrSh5GfqX3CsaguJ2nAj1GjSKJQ1Xc4AgYahoMXH6M_QlWHvOzUooPNbn04jn_39vyBwH7A8c26Tc0S5-6nKG8Nr_0r1_HNAu0mLmCOxmxpcagzxcfddKKhW9onED-j-Uy_atTZClPTEcorDaktDo1A05hZvzRNWnieHceUwtWp1FMPQZX4P7Yir3COups7eMShV91KLvQr7TpjbS99bjdAU8MPZqgybmGWheuFoMuxlp2WVrpQoa1XcJGg0eLPpwizQp-T1Pgxv-32rj6ElnSrdKVin5D5YG2hIgRyFwDywZuWxl3WCKBcvUgFCX4BZdhpjf1Y40x7lkmEthgTPZQD8XEvJ5pjWbunZ6WkeqtoZ49O0ifAQ29d-YK6VvW0VP3WGzvlOOTv1o42AE5Eks_OJTb2jriFX2XRYjsmzjWZk7Zw054zFfghNIvHEYDnAplmfjRxBTkLLOePsrhQaWhJlewBG-1igI31IgXbVrgRYnhHBdpdAvXyWK_Ih-GdKcBZKQ8JxR8ynRcVjJXmqrjO_EnJnbjMYemvwSNCAr-1w5VzUmJMMwWtpaJpXH07wmYhzBCo8r_Wzds_FC-57PfFl82uCuwu75F1wrf8CTckqLRR9QxoGJ6KZFV6NqbPAMf97WAMSRMYu60Jh5CZ1esKDmTi1TXEjKr8r6zWutarN1Z3xkt90a3azq7YULyGUYQRCWUpyYTkG0DZYuHsQeAF4MCDo2AFG8cIqZeBfpn-VKeP9FO65vGeQTrSQ3qecMT1utPXNQmu_XbFGTYEcOJgwv3J7EFjl8EUI2da5CjtPHcvYkj_HFqvn-an5KZwe27lUasYtObTEf0Az1_vUb2SbTO9JSZVxx3PMO5471hfaS4WWiZcAASVg831U7IYyb3ujDtCJjXoshoh-LEDrmEUWyJt5ifn9CQnON7RBrKrK5yPVF5mk91WbOET4ogwCWzZjC32v1-cP1yULplDB2rvmUQA3J_0dzNWoVoyWSoZt-iHyRDrLk4SwBfhYsttd4fPPaKq5rq4W8dJU_7qx4OeVWewpFA5ciG3A6QZkhdat37nkPPxd22QMBVab_CGZkpfltADhEV56VEwCE-YtfbKfXoY0MKH8pguwJ0bLUp9x1cFVPxlAQvEnY9vRJmD9JNwR3uoCMEumJHFDHDgy20jpKuOFG-ghCvg8FozN9Habjg1VXnsT1wgSs9z-WArMyFNinIlWArxiV7pIjqxJONHIArIOQdZUt29YDK5x_tKWIPQujmdBdvyUHqN5rsM8GXtSYW-sb8lfTliUF9FXReIhyP7e6YJNlF5f3Hn5KRdh0EEzR8s6SK7G108CZ0c4u0_Q_Q_L815QsE-qXD0w8-QyUmgI_3gNCCgaP0rvgNesWnujoEMEM6JM9n-xIQDNH0I5_STiksgSJcKvBzqojCT5eCr8x7VI1NWh01a1k6xDZRGP1g3gz1n0qjCD7GjLV7ZDpi5JnvYeLwPtjPZDATpw6bXbSOQ'
  if password:
    while True:
      try:
        key = decrypt_key(password)
        break
      except Exception as e:
        print(f"Wrong password. {e}")
        password = input("Enter password to use my reports service: ")

  else:
    while True:
      password = input("Enter password to use my reports service: ")
      try:
        key = decrypt_key(password)
        break
      except Exception as e:
        print(f"Wrong password. {e}")

  return recover(key, content)

def cls():
  """
  Clears the terminal screen. Works on both Windows and Linux.
  """
  if os.name == 'nt':  # For Windows
    _ = os.system('cls')
  else:  # For Linux and Mac
    _ = os.system('clear')


