import cloudfiles
import os
import marshmallow as mm
import sys


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
            value = "file://" + value
        cloudpath, file = os.path.split(value)
        cf = cloudfiles.CloudFiles(cloudpath)
        try:
            cf.put(".cftest", b"1")
            cf.delete(
                ".cftest",
            )
        except Exception as e:  # pragma: no cover
            raise mm.ValidationError(
                "%s cannot be written to cloud " % value
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
                cf.put(".cftest", b"0")
                cf.delete(".cftest")
            except:
                raise mm.ValidationError(
                    "{} is not a directory and you cannot write to it it".format(value)
                )


def validate_input_path(value):
    if "://" not in value:
        if not os.path.isfile(value):
            raise mm.ValidationError("%s is not a file" % value)
        else:
            try:
                with open(value) as f:
                    pass
            except Exception as value:
                raise mm.ValidationError("%s is not readable" % value)
    else:
        cloudpath, file = os.path.split(value)
        try:
            cf = cloudfiles.CloudFiles(cloudpath)
            if not cf.exists(file):
                raise mm.ValidationError("%s is not readable in cloud" % value)
        except ValueError:
            raise mm.ValidationError("%s is not readable in cloud" % value)


class InputDir(mm.fields.Str):
    """InputDir is  :class:`marshmallow.fields.Str` subclass which is a path to a
    a directory that exists and that the user can access
    (presently checked with os.access and cloudfiles.isdir())
    """

    def _validate(self, value):
        if "://" not in value:
            if not os.path.isdir(value):
                raise mm.ValidationError("%s is not a directory")

            if sys.platform == "win32":
                try:
                    x = list(os.scandir(value))
                except PermissionError:
                    raise mm.ValidationError("%s is not a readable directory" % value)
            else:
                if not os.access(value, os.R_OK):
                    raise mm.ValidationError("%s is not a readable directory" % value)
        else:
            try:
                cf = cloudfiles.CloudFiles(value)
                if not cf.isdir():
                    raise mm.ValidationError(
                        "%s is not a readable cloudpath directory" % value
                    )
            except:
                raise mm.ValidationError(
                    "%s is not a readable cloudpath directory" % value
                )


class InputFile(mm.fields.Str):
    """InputDile is a :class:`marshmallow.fields.Str` subclass which is a path to a
    file location which can be read by the user
    (presently passes os.path.isfile and os.access = R_OK)
    """

    def _validate(self, value):
        validate_input_path(value)
