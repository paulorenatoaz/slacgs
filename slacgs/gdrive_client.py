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

		response = self.drive_service.files().list(
			q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'", spaces='drive',
			fields='files(id)').execute()
		folders = response.get('files', [])

		if len(folders) > 0:
			return folders[0]['id']  # Return the ID of the first matching folder
		else:
			return None  # Return None if no folder with the given name is found

	def create_spreadsheet(self, name):

		spreadsheet = {
			'properties': {
				'title': name
			}
		}

		spreadsheet = self.sheets_service.spreadsheets().create(body=spreadsheet).execute()
		print(f"Spreadsheet with path '{self.get_spreadsheet_path(spreadsheet['spreadsheetId'])}' has been created.")
		return spreadsheet['spreadsheetId']

	def create_folder(self, folder_name, parent_folder_id=None):
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
		file = self.drive_service.files().get(fileId=file_id, fields='parents').execute()
		previous_parents = ",".join(file.get('parents'))
		file = self.drive_service.files().update(fileId=file_id, addParents=folder_id, removeParents=previous_parents,
		                                         fields='id, parents').execute()
		print(f"File with name '{file.get('name')}' has been moved to the folder with path "
		      f"'{self.get_folder_path(folder_id)}'.")

	def move_folder_to_another_folder(self, folder_id, new_parent_folder_id):
		# Retrieve the current parents of the folder
		folder = self.drive_service.files().get(fileId=folder_id, fields='parents').execute()
		previous_parents = ",".join(folder.get('parents'))

		# Move the folder to the new parent folder
		folder = self.drive_service.files().update(fileId=folder_id, addParents=new_parent_folder_id,
		                                      removeParents=previous_parents, fields='id, parents').execute()
		print(f"Folder with name '{folder.get('name')}' has been moved to the folder with name "
		      f"'{self.drive_service.files().get(fileId=new_parent_folder_id, fields='name').execute().get('name')}'.")

	def check_spreadsheet_existence(self, name):
		response = self.drive_service.files().list(
			q=f"name='{name}' and mimeType='application/vnd.google-apps.spreadsheet'", spaces='drive',
			fields='files(id)').execute()
		spreadsheets = response.get('files', [])

		if len(spreadsheets) > 0:
			return True  # A spreadsheet with the specified name exists
		else:
			return False  # No spreadsheet with the specified name exists

	def get_spreadsheet_id_by_name(self, name):
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
		self.drive_service.files().delete(fileId=spreadsheet_id).execute()
		print(f"Spreadsheet with ID '{spreadsheet_id}' has been deleted.")

	def check_folder_existence(self, report_folder_name):
		response = self.drive_service.files().list(
			q=f"name='{report_folder_name}' and mimeType='application/vnd.google-apps.folder'", spaces='drive',
			fields='files(id)').execute()
		folders = response.get('files', [])

		if len(folders) > 0:
			return True

	def get_root_folder_id(self):
		response = self.drive_service.files().get(fileId='root', fields='id').execute()
		return response.get('id')

	def get_folder_path(self, folder_id):
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
		permission = {
			'type': 'user',
			'role': 'writer',
			'emailAddress': self.gdrive_account_mail
		}

		self.drive_service.permissions().create(fileId=folder_id, body=permission).execute()
		print(f"Folder with ID '{folder_id}' has been shared with account (email: {self.gdrive_account_mail}).")

