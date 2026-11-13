from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
# SQLAlchemy 2.x is stricter about legacy ORM type annotations used by nuplan-devkit.
Base.__allow_unmapped__ = True
