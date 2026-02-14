import cloudfiles
import os
import marshmallow as mm
import sys
import secrets


class OutputFile(mm.fields.Str):
    """OutputFile :class:`marshmallow.fields.Str` subclass which is a path to a
       file location that can be written to by the current user
       (presently tested by opening a temporary file to that
       location)

    Parameters
    ----------

    Returns
    -------

    """

    def _validate(self, value):
        """

        Parameters
        ----------
        value : str
            filepath to validate you can write to that location

        Returns
        -------
        None

        Raises
        ------
        marshmallow.ValidationError
            If os.path.dirname cannot be applied, or if directory does not exist, or if you cannot write to that directory,
            or writing a temporary file there produces any crazy exception
        """
        if "://" not in value:
            dir, _ = os.path.split(value)
            if len(dir) == 0:
                value = './' + value 
            value = "file://" + value
        cloudpath, file = os.path.split(value)
        cf = cloudfiles.CloudFiles(cloudpath)
        try:
            hex = secrets.token_hex(16)
            cf.put(f".cftest_{hex}", b"1")
            cf.delete(f".cftest_{hex}")
        except Exception as e:  # pragma: no cover
            raise mm.ValidationError(
                f"{value} cannot be written to cloud "
            )  # pragma: no cover


class OutputDir(mm.fields.Str):
    """OutputDir is a :class:`marshmallow.fields.Str` subclass which is a path to
    a location where this module will write files.  Validation will check that
    the directory exists and create the directory if it is not present,
    and will fail validation if the directory cannot be created or cannot be
    written to.

    Parameters
    ==========
    mode: str
       mode to create directory
    *args:
      smae as passed to marshmallow.fields.Str
    **kwargs:
      same as passed to marshmallow.fields.Str
    """

    def __init__(self, *args, **kwargs):
        super(OutputDir, self).__init__(*args, **kwargs)

    def _validate(self, value):
        if "://" not in value:
            value = "file://" + value
        cf = cloudfiles.CloudFiles(value)

        if not cf.isdir():
            try:
                hex = secrets.token_hex(16)
                cf.put(f".cftest_{hex}", b"1")
                cf.delete(f".cftest_{hex}")
            except:
                raise mm.ValidationError(
                    "{} is not a directory and you cannot write to it it".format(value)
                )


def validate_input_path(value):
    if "://" not in value:
        if not os.path.isfile(value):
            raise mm.ValidationError(f"{value} is not a file")
        else:
            try:
                with open(value) as f:
                    pass
            except Exception as value:
                raise mm.ValidationError(f"{value} is not readable")
    else:
        cloudpath, file = os.path.split(value)
        try:
            cf = cloudfiles.CloudFiles(cloudpath)
            if not cf.exists(file):
                raise mm.ValidationError(f"{value} is not readable in cloud")
        except ValueError:
            raise mm.ValidationError(f"{value} is not readable in cloud")


class InputDir(mm.fields.Str):
    """InputDir is  :class:`marshmallow.fields.Str` subclass which is a path to a
    a directory that exists and that the user can access
    (presently checked with os.access and cloudfiles.isdir())
    """

    def _validate(self, value):
        if "://" not in value:
            if not os.path.isdir(value):
                raise mm.ValidationError(f"{value} is not a directory")

            if sys.platform == "win32":
                try:
                    x = list(os.scandir(value))
                except PermissionError:
                    raise mm.ValidationError(f"{value} is not a readable directory")
            else:
                if not os.access(value, os.R_OK):
                    raise mm.ValidationError(f"{value} is not a readable directory")
        else:
            try:
                cf = cloudfiles.CloudFiles(value)
                if not cf.isdir():
                    raise mm.ValidationError(
                        f"{value} is not a readable cloudpath directory"
                    )
            except:
                raise mm.ValidationError(
                    f"{value} is not a readable cloudpath directory"
                )


class InputFile(mm.fields.Str):
    """InputDile is a :class:`marshmallow.fields.Str` subclass which is a path to a
    file location which can be read by the user
    (presently passes os.path.isfile and os.access = R_OK)
    """

    def _validate(self, value):
        validate_input_path(value)
