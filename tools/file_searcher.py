import glob
import pathlib


class FileSearcher:
    def __init__(self, folder, types, *, is_recursive=False):
        self.folder = folder
        self.types = types
        self.photos_list = []
        self.is_recursive = is_recursive

        path = str(pathlib.Path(self.folder + ("/**/" if is_recursive else "") + "/*.*"))
        for file in glob.glob(path, recursive=self.is_recursive):
            file_path = pathlib.Path(file)

            file_name = file_path.name
            file_extension = file_path.suffix[1:].casefold()

            if file_extension in self.types:
                result = {"path": file, "filename": file_name}
                self.photos_list.append(result)

    def search_by_keyword(self, keyword) -> list:
        list_filter = list(filter(lambda x: keyword in x[1], self.photos_list))
        list_files = [i["filename"] for i in list_filter]

        return list_files

    @property
    def paths(self):
        return [i["path"] for i in self.photos_list]

    @property
    def files(self):
        return [i["filename"] for i in self.photos_list]

    @property
    def photos(self):
        return self.photos_list
