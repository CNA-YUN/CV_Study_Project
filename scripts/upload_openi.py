import openi
from openi import openi_upload_file
from _init_ import DATA_DIR

openi.login(token='0ec626ba6fdac57e828ec0b3bb67601b544b2483')

openi_upload_file(repo_id="zhangyun_cqupt/OxfordPet",
                  file_or_folder_path=DATA_DIR / 'OxfordPet',
                  max_workers=50)
