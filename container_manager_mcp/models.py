from pydantic import BaseModel, ConfigDict


class BaseDictModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    def __getitem__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        if self.model_extra and item in self.model_extra:
            return self.model_extra[item]
        raise KeyError(item)

    def __contains__(self, item):
        return hasattr(self, item) or (
            bool(self.model_extra) and item in self.model_extra
        )

    def get(self, item, default=None):
        try:
            return self[item]
        except KeyError:
            return default


class CommandResult(BaseDictModel):
    success: bool
    message: str | None = None
    error: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    command: str | None = None


class ImageInfo(BaseDictModel):
    id: str
    repository: str
    tag: str
    created: str
    size: str


class ContainerInfo(BaseDictModel):
    id: str
    name: str
    image: str
    status: str
    ports: str
    created: str


class VolumeInfo(BaseDictModel):
    name: str
    driver: str
    mountpoint: str
    created: str | None = None


class NetworkInfo(BaseDictModel):
    id: str
    name: str
    driver: str
    scope: str
