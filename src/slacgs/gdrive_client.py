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

	def create_folder(folder_name, parent_folder_id=None):
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
		file = self.drive_service.files().update(fileId=file_id, addParents=folder_id, removeParents=previous_parents, fields='id, parents').execute()
