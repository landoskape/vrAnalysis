from pathlib import Path
from typing import Union, List


class Figure_Saver:
    """
    Class for saving figures
    RH 2022

    stolen with permission from Rich Hakim's repo:
    https://github.com/RichieHakim/basic_neural_processing_modules/tree/main
    """

    def __init__(
        self,
        dir_save: str = None,
        format_save: list = ["png", "svg"],
        kwargs_savefig: dict = {
            "bbox_inches": "tight",
            "pad_inches": 0.1,
            "transparent": True,
            "dpi": 300,
        },
        overwrite: bool = False,
        mkdir: bool = True,
        verbose: int = 1,
        enabled: bool = True,
    ):
        """
        Initializes Figure_Saver object

        Args:
            dir_save (str):
                Directory to save the figure. Used if path_config is None.
                Must be specified if path_config is None.
            format (list of str):
                Format(s) to save the figure. Default is 'png'.
                Others: ['png', 'svg', 'eps', 'pdf']
            overwrite (bool):
                If True, then overwrite the file if it exists.
            kwargs_savefig (dict):
                Keyword arguments to pass to fig.savefig().
            verbose (int):
                Verbosity level.
                0: No output.
                1: Warning.
                2: All info.
            enabled (bool):
                If False, then the save() method will not save the figure.
        """
        self.dir_save = str(Path(dir_save).resolve().absolute()) if dir_save is not None else None

        assert isinstance(format_save, list), "RH ERROR: format_save must be a list of strings"
        assert all([isinstance(f, str) for f in format_save]), "RH ERROR: format_save must be a list of strings"
        self.format_save = format_save

        assert isinstance(kwargs_savefig, dict), "RH ERROR: kwargs_savefig must be a dictionary"
        self.kwargs_savefig = kwargs_savefig

        self.overwrite = overwrite
        self.mkdir = mkdir
        self.verbose = verbose
        self.enabled = enabled

    def save(
        self,
        fig,
        name_file: Union[str, List[str]] = None,
        path_save: str = None,
        dir_save: str = None,
        overwrite: bool = None,
    ):
        """
        Save the figures.

        Args:
            fig (matplotlib.figure.Figure):
                Figure to save.
            name_file (Union[str, List[str]):
                Name of the file to save.\n
                If None, then the title of the figure is used.\n
                Path will be dir_save / name_file.\n
                If a list of strings, then elements [:-1] will be subdirectories
                and the last element will be the file name.
            path_save (str):
                Path to save the figure.
                Should not contain suffix.
                If None, then the dir_save must be specified here or in
                 the initialization and name_file must be specified.
            dir_save (str):
                Directory to save the figure. If None, then the directory
                 specified in the initialization is used.
            overwrite (bool):
                If True, then overwrite the file if it exists. If None, then the
                value specified in the initialization is used.
        """
        if not self.enabled:
            print("RH Warning: Figure_Saver is disabled. Not saving the figure.") if self.verbose > 1 else None
            return None

        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure), "RH ERROR: fig must be a matplotlib.figure.Figure"

        ## Get path_save
        if path_save is not None:
            assert len(Path(path_save).suffix) == 0, "RH ERROR: path_save must not contain suffix"
            path_save = [str(Path(path_save).resolve()) + "." + f for f in self.format_save]
        else:
            assert (dir_save is not None) or (self.dir_save is not None), "RH ERROR: dir_save must be specified if path_save is None"
            assert name_file is not None, "RH ERROR: name_file must be specified if path_save is None"

            ## Get dir_save
            dir_save = self.dir_save if dir_save is None else str(Path(dir_save).resolve())

            ## Get figure title
            if name_file is None:
                titles = [a.get_title() for a in fig.get_axes() if a.get_title() != ""]
                name_file = ".".join(titles)
            if isinstance(name_file, list):
                assert all([isinstance(f, str) for f in name_file]), "RH ERROR: name_file must be a string or a list of strings"
                assert len(name_file) > 1, "RH ERROR: If name_file is a list, then it must have more than one element"
                dir_save = str(Path(dir_save) / Path(*name_file[:-1]))
                name_file = name_file[-1]
            path_save = [str(Path(dir_save) / (name_file + "." + f)) for f in self.format_save]

        ## Make directory
        if self.mkdir:
            Path(path_save[0]).parent.mkdir(parents=True, exist_ok=True)

        path_save = [path_save] if not isinstance(path_save, list) else path_save

        ## Check overwrite
        overwrite = self.overwrite if overwrite is None else overwrite

        ## Save figure
        for path, form in zip(path_save, self.format_save):
            if Path(path).exists():
                if overwrite:
                    print(f"RH Warning: Overwriting file. File: {path} already exists.") if self.verbose > 0 else None
                else:
                    print(f"RH Warning: Not saving anything. File exists and overwrite==False. {path} already exists.") if self.verbose > 0 else None
                    return None
            print(f"FR: Saving figure {path} as format(s): {form}") if self.verbose > 1 else None
            fig.savefig(path, format=form, **self.kwargs_savefig)

    def save_batch(
        self,
        figs,
        dir_save: str = None,
        names_files: str = None,
    ):
        """
        Save all figures in a list.

        Args:
            figs (list of matplotlib.figure.Figure):
                Figures to save.
            dir_save (str):
                Directory to save the figure. If None, then the directory
                 specified in the initialization is used.
            name_file (str):
                Name of the file to save. If None, then the name of
                the figure is used.
        """
        import matplotlib.pyplot as plt

        assert isinstance(figs, list), "RH ERROR: figs must be a list of matplotlib.figure.Figure"
        assert all([isinstance(fig, plt.Figure) for fig in figs]), "RH ERROR: figs must be a list of matplotlib.figure.Figure"

        ## Get dir_save
        dir_save = self.dir_save if dir_save is None else str(Path(dir_save).resolve())

        for fig, name_file in zip(figs, names_files):
            self.save(fig, name_file=name_file, dir_save=dir_save)

    def __call__(
        self,
        fig,
        name_file: str = None,
        path_save: str = None,
        dir_save: str = None,
        overwrite: bool = None,
    ):
        """
        Calls save() method.
        """
        self.save(fig, path_save=path_save, name_file=name_file, dir_save=dir_save, overwrite=overwrite)

    def __repr__(self):
        return f"Figure_Saver(dir_save={self.dir_save}, format={self.format_save}, overwrite={self.overwrite}, kwargs_savefig={self.kwargs_savefig}, verbose={self.verbose})"
