from google.oauth2 import service_account
from googleapiclient.discovery import build
import os


class GdriveClient:
	"""perform operations on Google Drive."""

	def __init__(self, key_path):

		"""Constructor for GdriveClient class.

		:param key_path: path to Key file for accessing Google Sheets API via Service Account Credentials.
		:type key_path: str

		:raises ValueError: if key_path is not a string.
		:raises FileNotFoundError: if key_path is not a valid path.
		"""

		if not isinstance(key_path, str):
			raise ValueError('key_path must be a string.')

		if not os.path.exists(key_path):
			print('key_path: ',key_path)
			raise FileNotFoundError('key_path is not a valid path.')

		SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']
		credentials = service_account.Credentials.from_service_account_file(key_path, scopes=SCOPES)
		self.drive_service = build('drive', 'v3', credentials=credentials)
		self.sheets_service = build('sheets', 'v4', credentials=credentials)
		self.gdrive_account_mail = None

	def get_folder_id_by_name(self, folder_name):

		"""Get the ID of a folder by its name. If no folder with the given name is found, return None.

		:param folder_name: name of the folder.
		:type folder_name: str

		:returns: ID of the folder with the given name.
		:rtype: str or None

		:raises ValueError: if folder_name is not a string.

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

	def create_spreadsheet(self, name):
		"""Create a new spreadsheet with the given name.

		:param name: name of the spreadsheet.
		:type name: str

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
		print(f"Spreadsheet with path '{self.get_spreadsheet_path_by_id(spreadsheet['spreadsheetId'])}' has been created.")
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

	def create_folder(self, folder_name, parent_folder_id=None):
		"""Create a new folder with the given name. If parent_folder_id is not None, the folder will be created inside the folder with the given ID.

		:param folder_name: name of the folder.
		:type folder_name: str

		:param parent_folder_id: ID of the parent folder.
		:type parent_folder_id: str

		:returns: ID of the created folder.
		:rtype: str

		:raises ValueError: if folder_name or parent_folder_id is not a string.

		"""

		if not isinstance(folder_name, str):
			raise ValueError('folder_name must be a string.')

		if parent_folder_id is not None and not isinstance(parent_folder_id, str):
			raise ValueError('parent_folder_id must be a string or None.')


		folder_metadata = {
			'name': folder_name,
			'mimeType': 'application/vnd.google-apps.folder'
		}
		if parent_folder_id:
			folder_metadata['parents'] = [parent_folder_id]

		folder = self.drive_service.files().create(body=folder_metadata, fields='id').execute()
		print(f"Folder with path '{self.get_folder_path(folder['id'])}' has been created.")
		return folder['id']

	def move_file_to_folder(self, file_id, folder_id):
		"""Move a file to a folder.

		:param file_id: ID of the file to be moved.
		:type file_id: str

		:param folder_id: ID of the folder to which the file will be moved.
		:type folder_id: str

		:raises ValueError: if file_id or folder_id is not a string.

		"""

		file = self.drive_service.files().get(fileId=file_id, fields='parents').execute()
		previous_parents = ",".join(file.get('parents'))
		file = self.drive_service.files().update(fileId=file_id, addParents=folder_id, removeParents=previous_parents,
		                                         fields='id, parents').execute()
		print(f"File has been moved to the folder with path "
		      f"'{self.get_folder_path(folder_id)}'.")

	def move_folder_to_another_folder(self, folder_id, new_parent_folder_id):
		"""Move a folder to another folder.

		:param folder_id: ID of the folder to be moved.
		:type folder_id: str

		:param new_parent_folder_id: ID of the folder to which the folder will be moved.
		:type new_parent_folder_id: str

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
			return spreadsheets[0]['id'] # Return the ID of the first matching spreadsheet
		else:
			return None # No spreadsheet with the specified name exists

	def delete_spreadsheet(self, spreadsheet_id):
		"""Delete a spreadsheet.

		:param spreadsheet_id: ID of the spreadsheet to be deleted.
		:type spreadsheet_id: str

		:raises ValueError: if spreadsheet_id is not a string.

		"""

		if not isinstance(spreadsheet_id, str):
			raise ValueError('spreadsheet_id must be a string.')


		self.drive_service.files().delete(fileId=spreadsheet_id).execute()
		print(f"Spreadsheet with ID '{spreadsheet_id}' has been deleted.")

	def check_folder_existence(self, report_folder_name):
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

	def share_folder_with_gdrive_account(self, folder_id):
		"""Share a folder with the GDrive account.

		:param folder_id: ID of the folder.
		:type folder_id: str

		:raises ValueError: if folder_id is not a string.

		"""

		if not isinstance(folder_id, str):
			raise ValueError('folder_id must be a string.')


		permission = {
			'type': 'user',
			'role': 'writer',
			'emailAddress': self.gdrive_account_mail
		}

		self.drive_service.permissions().create(fileId=folder_id, body=permission).execute()
		print(f"Folder with path '{self.get_folder_path(folder_id)}' has been shared with the GDrive account with "
		      f"email address '{self.gdrive_account_mail}'.")

	def delete_folder(self, folder_id):
		"""Delete a folder.

		:param folder_id: ID of the folder.
		:type folder_id: str

		:raises ValueError: if folder_id is not a string.

		"""

		if not isinstance(folder_id, str):
			raise ValueError('folder_id must be a string.')

		self.drive_service.files().delete(fileId=folder_id).execute()
		print(f"Folder with path '{self.get_folder_path(folder_id)}' has been deleted.")

	def delete_file(self, file_id):
		self.drive_service.files().delete(fileId=file_id).execute()
		print(f"File with ID '{file_id}' has been deleted.")
