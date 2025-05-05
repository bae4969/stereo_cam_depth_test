import os


class MyCommon:
    _base_path: str
    _src_path: str
    _dst_path: str

    def __init__(self, base_path:str, src_path:str = None, dst_path:str = None):
        self._base_path = base_path
        self._src_path = src_path
        self._dst_path = dst_path

    def GetSrcDirPath(self):
        return os.path.join(self._base_path, self._src_path)

    def GetDstDirPath(self):
        return os.path.join(self._base_path, self._dst_path)

    def GetSrcFileList(self, ext: str):
        return [
            f
            for f in os.listdir(self.GetSrcDirPath())
            if f.endswith(ext)
        ]

    def GetSrcFilePath(self, filename: str):
        return os.path.join(self.GetSrcDirPath(), filename)

    def GetDstFilePath(self, filename: str):
        return os.path.join(self.GetDstDirPath(), filename)
