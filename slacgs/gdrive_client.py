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
			raise FileNotFoundError('key_path is not a valid path.')

		SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']
		credentials = service_account.Credentials.from_service_account_file(key_path, scopes=SCOPES)
		self.drive_service = build('drive', 'v3', credentials=credentials)
		self.sheets_service = build('sheets', 'v4', credentials=credentials)

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
		return spreadsheet['spreadsheetId']

	def create_folder(self, folder_name, parent_folder_id=None):
		folder_metadata = {
			'name': folder_name,
			'mimeType': 'application/vnd.google-apps.folder'
		}
		if parent_folder_id:
			folder_metadata['parents'] = [parent_folder_id]

		folder = self.drive_service.files().create(body=folder_metadata, fields='id').execute()
		return folder['id']

	def move_file_to_folder(self, file_id, folder_id):
		file = self.drive_service.files().get(fileId=file_id, fields='parents').execute()
		previous_parents = ",".join(file.get('parents'))
		file = self.drive_service.files().update(fileId=file_id, addParents=folder_id, removeParents=previous_parents,
		                                         fields='id, parents').execute()

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


