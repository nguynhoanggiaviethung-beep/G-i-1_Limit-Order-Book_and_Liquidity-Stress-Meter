"""Database integration is intentionally optional for the Parquet MVP."""


def get_database():
    raise NotImplementedError("Parquet is the MVP storage backend; configure a database when multi-user access is needed")
