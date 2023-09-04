import io
import os
import re
from googleapiclient.discovery import Resource
from googleapiclient.http import MediaIoBaseUpload

from .utils import report_service_conf, set_report_service_conf


class GdriveClient:
	"""perform operations on Google Drive."""

	def __init__(self, drive_service=None, spreadsheet_service=None, gdrive_account_email=None):

		"""Constructor for GdriveClient class.

		Parameters:
			drive_service (googleapiclient.discovery.Resource): Google Drive API Resource object.
			spreadsheet_service (googleapiclient.discovery.Resource): Google Sheets API Resource object.
			gdrive_account_email (str): email of the Google account to be used to share reports folder.

		Raises:
			TypeError:
				if drive_service is not a googleapiclient.discovery.Resource object;
				if sheets_service is not a googleapiclient.discovery.Resource object;
				if gdrive_account_email is not a string.

			ValueError:
				if gdrive_account_email is not a valid email address.

		"""

		if drive_service and not isinstance(drive_service, Resource):
			raise TypeError('drive_service must be a googleapiclient.discovery.Resource object.')

		if spreadsheet_service and not isinstance(spreadsheet_service, Resource):
			raise TypeError('sheets_service must be a googleapiclient.discovery.Resource object.')

		if gdrive_account_email:
			if not isinstance(gdrive_account_email, str):
				raise TypeError('gdrive_account_email must be a string.')

			if not re.match(r"[^@]+@[^@]+\.[^@]+", gdrive_account_email):
				raise ValueError('gdrive_account_email must be a valid email address.')

		else:
			if report_service_conf['user_email']:
				gdrive_account_email = report_service_conf['user_email']
			else:
				while True:
					gdrive_account_email = input(
						"Please enter your google account email address so I can share results with you\n(if you don't have a google account, create one at https://accounts.google.com/signup)\ngoogle account email:")
					if re.match(r"[^@]+@[^@]+\.[^@]+", gdrive_account_email):
						print("Valid email address!")
						break
					else:
						print("Invalid email address. Please try again.")

		## If drive_service and sheets_service are not provided, try to get them from report_service. If report_service is not running, start it.
		if not drive_service and not spreadsheet_service:
			if not report_service_conf['drive_service'] or not report_service_conf['spreadsheet_service']:
				set_report_service_conf()

			drive_service = report_service_conf['drive_service']
			spreadsheet_service = report_service_conf['spreadsheet_service']

		self.drive_service = drive_service
		self.sheets_service = spreadsheet_service
		self.gdrive_account_email = gdrive_account_email

	def get_folder_id_by_name(self, folder_name):

		"""Get the ID of a folder by its name. If no folder with the given name is found, return None.

		:param folder_name: name of the folder.
		:type folder_name: str

		:returns: ID of the folder with the given name.
		:rtype: str or None

		:raises ValueError: if folder_name is not a string.

		Parameters:
			folder_name (str): name of the folder.

		Returns:

		"""

		if not isinstance(folder_name, str):
			raise ValueError('folder_name must be a string.')

		response = self.drive_service.files().list(
			q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'", spaces='drive',
			fields='files(id)').execute()
		folders = response.get('files', [])

		if len(folders) > 0:
			return folders[0]['id']  # Return the ID of the first matching folder
		else:
			return None  # Return None if no folder with the given name is found

	def create_spreadsheet(self, name, verbose=True):
		"""Create a new spreadsheet with the given name.

		:param name: name of the spreadsheet.
		:type name: str

		:param verbose: if True, print a message after creating the spreadsheet.
		:type verbose: bool

		:returns: ID of the created spreadsheet.
		:rtype: str



		:raises ValueError: if name is not a string.

		"""

		if not isinstance(name, str):
			raise ValueError('name must be a string.')

		spreadsheet = {
			'properties': {
				'title': name
			}
		}

		spreadsheet = self.sheets_service.spreadsheets().create(body=spreadsheet).execute()
		if verbose:
			print(
				f"Spreadsheet with path '{self.get_spreadsheet_path_by_id(spreadsheet['spreadsheetId'])}' has been created.")

		return spreadsheet['spreadsheetId']

	def get_spreadsheet_path_by_id(self, spreadsheet_id):
		"""Get the path of a spreadsheet by its ID.

		:param spreadsheet_id: ID of the spreadsheet.
		:type spreadsheet_id: str

		:returns: path of the spreadsheet with the given ID.
		:rtype: str

		:raises ValueError: if spreadsheet_id is not a string.

		"""

		if not isinstance(spreadsheet_id, str):
			raise ValueError('spreadsheet_id must be a string.')

		spreadsheet = self.sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
		return spreadsheet['properties']['title']

	def create_folder(self, folder_name, parent_folder_id=None, verbose=True):
		"""Create a new folder with the given name. If parent_folder_id is not None, the folder will be created inside the folder with the given ID. If folder path already exists, return the ID of the existing folder.

		:param folder_name: name of the folder.
		:type folder_name: str

		:param parent_folder_id: ID of the parent folder.
		:type parent_folder_id: str

		:param verbose: whether to print the path of the created folder.
		:type verbose: bool

		:returns: ID of the created folder.
		:rtype: str

		:raises ValueError: if folder_name or parent_folder_id is not a string.

		"""

		if not isinstance(folder_name, str):
			raise ValueError('folder_name must be a string.')

		if parent_folder_id is not None and not isinstance(parent_folder_id, str):
			raise ValueError('parent_folder_id must be a string or None.')

		if parent_folder_id is not None:
			parent_path = self.get_folder_path(parent_folder_id)
			if self.folder_exists_by_path(f"{parent_path}/{folder_name}"):
				return self.get_folder_id_by_path(f"{parent_path}/{folder_name}")
		else:
			if self.folder_exists_by_path(folder_name):
				return self.get_folder_id_by_path(folder_name)

		folder_metadata = {
			'name': folder_name,
			'mimeType': 'application/vnd.google-apps.folder'
		}
		if parent_folder_id:
			folder_metadata['parents'] = [parent_folder_id]

		folder = self.drive_service.files().create(body=folder_metadata, fields='id').execute()
		if verbose:
			print(f"Folder with path '{self.get_folder_path(folder['id'])}' has been created.")

		return folder['id']

	def move_file_to_folder(self, file_id, folder_id, verbose=True):
		"""Move a file to a folder.

		:param file_id: ID of the file to be moved.
		:type file_id: str

		:param folder_id: ID of the folder to which the file will be moved.
		:type folder_id: str

		:param verbose: whether to print messages about the operation status.
		:type verbose: bool

		:raises ValueError: if file_id or folder_id is not a string.

		"""
		file_old_path = self.get_spreadsheet_path_by_id(file_id)

		file = self.drive_service.files().get(fileId=file_id, fields='parents').execute()
		previous_parents = ",".join(file.get('parents'))

		self.drive_service.files().update(fileId=file_id, addParents=folder_id, removeParents=previous_parents,
		                                  fields='id, parents').execute()
		if verbose:
			print(
				f"Spreadsheet with path '{file_old_path}' has been moved to the folder with path '{self.get_folder_path(folder_id)}'.")

	def move_folder_to_another_folder(self, folder_id, new_parent_folder_id, verbose=True):
		"""Move a folder to another folder.

		:param folder_id: ID of the folder to be moved.
		:type folder_id: str

		:param new_parent_folder_id: ID of the folder to which the folder will be moved.
		:type new_parent_folder_id: str

		:param verbose: whether to print messages about the operation status.
		:type verbose: bool

		:returns: None
		:rtype: None

		:raises ValueError: if folder_id or new_parent_folder_id is not a string.

		"""

		if not isinstance(folder_id, str):
			raise ValueError('folder_id must be a string.')

		if not isinstance(new_parent_folder_id, str):
			raise ValueError('new_parent_folder_id must be a string.')

		# Retrieve the current parents of the folder
		folder = self.drive_service.files().get(fileId=folder_id, fields='parents').execute()
		previous_parents = ",".join(folder.get('parents'))

		# Move the folder to the new parent folder
		folder = self.drive_service.files().update(fileId=folder_id, addParents=new_parent_folder_id,
		                                           removeParents=previous_parents, fields='id, parents').execute()
		if verbose:
			print(f"Folder with name '{folder.get('name')}' has been moved to the folder with name "
			      f"'{self.drive_service.files().get(fileId=new_parent_folder_id, fields='name').execute().get('name')}'.")

	def check_spreadsheet_existence(self, name):
		"""Check if a spreadsheet with the given name exists.

		:param name: name of the spreadsheet.
		:type name: str

		:returns: True if a spreadsheet with the given name exists, False otherwise.
		:rtype: bool

		:raises ValueError: if name is not a string.

		"""

		if not isinstance(name, str):
			raise ValueError('name must be a string.')

		response = self.drive_service.files().list(
			q=f"name='{name}' and mimeType='application/vnd.google-apps.spreadsheet'", spaces='drive',
			fields='files(id)').execute()
		spreadsheets = response.get('files', [])

		if len(spreadsheets) > 0:
			return True  # A spreadsheet with the specified name exists
		else:
			return False  # No spreadsheet with the specified name exists

	def get_spreadsheet_id_by_name(self, name):
		"""Get the ID of a spreadsheet with the given name.

		:param name: name of the spreadsheet.
		:type name: str

		:returns: ID of the spreadsheet with the given name.
		:rtype: str

		:raises ValueError: if name is not a string.

		"""

		if not isinstance(name, str):
			raise ValueError('name must be a string.')

		response = self.drive_service.files().list(
			q=f"name='{name}' and mimeType='application/vnd.google-apps.spreadsheet'",
			spaces='drive',
			fields='files(id)').execute()
		spreadsheets = response.get('files', [])
		if len(spreadsheets) > 0:
			return spreadsheets[0]['id']  # Return the ID of the first matching spreadsheet
		else:
			return None  # No spreadsheet with the specified name exists

	def delete_spreadsheet(self, spreadsheet_id):
		"""Delete a spreadsheet.

		:param spreadsheet_id: ID of the spreadsheet to be deleted.
		:type spreadsheet_id: str

		:raises ValueError: if spreadsheet_id is not a string.

		"""

		if not isinstance(spreadsheet_id, str):
			raise ValueError('spreadsheet_id must be a string.')

		path = self.get_spreadsheet_path_by_id(spreadsheet_id)

		self.drive_service.files().delete(fileId=spreadsheet_id).execute()
		print(f"Spreadsheet with path '{path}' has been deleted.")

	def folder_exists(self, report_folder_name):
		"""Check if a folder with the given name exists.

		:param report_folder_name: name of the folder.
		:type report_folder_name: str

		:returns: True if a folder with the given name exists, False otherwise.
		:rtype: bool

		:raises ValueError: if report_folder_name is not a string.

		"""

		if not isinstance(report_folder_name, str):
			raise ValueError('report_folder_name must be a string.')

		response = self.drive_service.files().list(
			q=f"name='{report_folder_name}' and mimeType='application/vnd.google-apps.folder'", spaces='drive',
			fields='files(id)').execute()
		folders = response.get('files', [])

		if len(folders) > 0:
			return True

	def get_root_folder_id(self):
		"""Get the ID of the root folder.

		:returns: ID of the root folder.
		:rtype: str

		"""

		response = self.drive_service.files().get(fileId='root', fields='id').execute()
		return response.get('id')

	def get_folder_path(self, folder_id):
		"""Get the path of a folder.

		:param folder_id: ID of the folder.
		:type folder_id: str

		:returns: path of the folder.
		:rtype: str

		:raises ValueError: if folder_id is not a string.

		"""

		if not isinstance(folder_id, str):
			raise ValueError('folder_id must be a string.')

		folder_path = []
		while folder_id != 'root':
			folder = self.drive_service.files().get(fileId=folder_id, fields='id, name, parents').execute()
			folder_name = folder.get('name')
			folder_path.insert(0, folder_name)
			parents = folder.get('parents')
			if parents:
				folder_id = parents[0]
			else:
				break
		return '/'.join(folder_path)

	def share_folder_with_gdrive_account(self, folder_id, verbose=True):
		"""Share a folder with the GDrive account.

		:param folder_id: ID of the folder.
		:type folder_id: str
		
		:param verbose: whether to print messages or not.
		:type verbose: bool
	
		:return: None
		:rtype: None

		:raises ValueError: if folder_id is not a string.
		
		"""

		if not isinstance(folder_id, str):
			raise ValueError('folder_id must be a string.')

		permission = {
			'type': 'user',
			'role': 'writer',
			'emailAddress': self.gdrive_account_email
		}

		self.drive_service.permissions().create(fileId=folder_id, body=permission).execute()

		if verbose:
			print(f"Folder with path '{self.get_folder_path(folder_id)}' has been shared with the GDrive account with "
			      f"email address '{self.gdrive_account_email}'.")
			print(f"link: https://drive.google.com/drive/folders/{folder_id}")

	def delete_folder(self, folder_id):
		"""Delete a folder.

		:param folder_id: ID of the folder.
		:type folder_id: str

		:raises ValueError: if folder_id is not a string.

		"""
		path = self.get_folder_path(folder_id)
		if not isinstance(folder_id, str):
			raise ValueError('folder_id must be a string.')

		self.drive_service.files().delete(fileId=folder_id).execute()
		print(f"Folder with path '{path}' has been deleted.")

	def delete_file(self, file_id):
		"""Delete a file.

		:param file_id: ID of the file.
		:type file_id: str

		"""

		self.drive_service.files().delete(fileId=file_id).execute()
		print(f"File with ID '{file_id}' has been deleted.")

	def get_file_id_by_path(self, file_path):
		"""Get the ID of a file.

		:param file_path: path of the file.
		:type file_path: str

		:returns: ID of the file.
		:rtype: str

		:raises ValueError: if file_path is not a string.

		"""

		if not isinstance(file_path, str):
			raise ValueError('file_path must be a string.')

		file_path = file_path.split('/')
		file_path = [file for file in file_path if file != '']

		file_id = 'root'
		for file_name in file_path:
			response = self.drive_service.files().list(
				q=f"name='{file_name}' and mimeType!='application/vnd.google-apps.folder' and '{file_id}' in parents",
				spaces='drive', fields='files(id)').execute()
			files = response.get('files', [])
			if len(files) > 0:
				file_id = files[0]['id']
			else:
				return None
		return file_id

	##make a function that check if folder with given path exists and return True if it does and False if it doesn't
	def folder_exists_by_path(self, folder_path):
		"""Check if a folder with the given path exists.

		Parameters:
			folder_path (str): The path to the folder you want to check.

		Returns:
			True if the folder exists, False otherwise.

		Raises:
			ValueError: If folder_path is not a string.


		"""

		if not isinstance(folder_path, str):
			raise ValueError('folder_path must be a string.')

		folder_path = folder_path.split('/')
		folder_path = [folder for folder in folder_path if folder != '']

		folder_id = 'root'
		for folder_name in folder_path:
			response = self.drive_service.files().list(
				q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and parents='{folder_id}'",
				spaces='drive', fields='files(id)').execute()
			folders = response.get('files', [])

			if len(folders) > 0:
				folder_id = folders[0].get('id')
			else:
				return False
		return True

	##make a function to get folder id by path
	def get_folder_id_by_path(self, folder_path):
		"""Get the ID of a folder by its path.

		Parameters:
			folder_path (str): The path to the folder you want to get the ID of.

		Returns:
			The ID of the folder.

		Raises:
			ValueError: If folder_path is not a string.

		"""

		if not isinstance(folder_path, str):
			raise ValueError('folder_path must be a string.')

		folder_path = folder_path.split('/')
		folder_path = [folder for folder in folder_path if folder != '']

		folder_id = 'root'
		for folder_name in folder_path:
			response = self.drive_service.files().list(
				q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and parents='{folder_id}'",
				spaces='drive', fields='files(id)').execute()
			folders = response.get('files', [])

			if len(folders) > 0:
				folder_id = folders[0].get('id')
			else:
				return None
		return folder_id

	##make a function to check if file with given path exists and return True if it does and False if it doesn't
	def file_exists_by_path(self, file_path):
		"""Check if a file with the given path exists.

		Parameters:
			file_path (str): The path to the file you want to check.

		Returns:
			True if the file exists, False otherwise.

		Raises:
			ValueError: If file_path is not a string.

		"""

		if not isinstance(file_path, str):
			raise ValueError('file_path must be a string.')

		file_name = os.path.basename(file_path)
		folder_path = os.path.dirname(file_path)

		folder_id = self.get_folder_id_by_path(folder_path)
		if folder_id is None:
			return False

		response = self.drive_service.files().list(
			q=f"name='{file_name}' and mimeType!='application/vnd.google-apps.folder' and parents='{folder_id}'",
			spaces='drive', fields='files(id)').execute()
		files = response.get('files', [])

		if len(files) > 0:
			return True
		else:
			return False

	def file_exists_in_folder(self, file_name, folder_id):
		"""Check if a file with the given name exists in the given folder.

		Parameters:
			file_name (str): The name of the file you want to check.
			folder_id (str): The ID of the folder you want to check.

		Returns:
			True if the file exists, False otherwise.

		Raises:
			ValueError: If file_name or folder_id is not a string.

		"""

		if not isinstance(file_name, str):
			raise ValueError('file_name must be a string.')
		if not isinstance(folder_id, str):
			raise ValueError('folder_id must be a string.')

		response = self.drive_service.files().list(
			q=f"name='{file_name}' and mimeType!='application/vnd.google-apps.folder' and parents='{folder_id}'",
			spaces='drive', fields='files(id)').execute()
		files = response.get('files', [])

		if len(files) > 0:
			return True
		else:
			return False

	def upload_file_to_drive(self, file_path, folder_id, verbose=True):
		"""
		Uploads a file to a Google Drive folder.

		Args:
				file_path (str): The path to the file you want to upload.
				folder_id (str): The ID of the Google Drive folder where you want to upload the file.

		"""

		# Prepare the file metadata
		file_name = os.path.basename(file_path)
		file_metadata = {
			'name': file_name,
			'parents': [folder_id]
		}

		# Upload the file
		media = MediaIoBaseUpload(io.BytesIO(open(file_path, 'rb').read()), mimetype='application/octet-stream',
		                          resumable=True)

		try:
			if self.file_exists_in_folder(file_name, folder_id):
				file_id = self.get_file_id_by_path(file_path)
			else:
				uploaded_file = self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
				file_id = uploaded_file.get('id')

			if verbose:
				print(f"File with path '{file_path}' has been uploaded to folder: "
				      f"https://drive.google.com/drive/folders/{folder_id}.")
				print('link to file: https://drive.google.com/file/d/' + file_id)
		except Exception as e:
			print(
				'failed to upload file with path: ' + file_path + ' to folder: https://drive.google.com/drive/folders/' + folder_id)

