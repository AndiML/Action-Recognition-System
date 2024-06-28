"""Represents a module that contains the base classes for commands and their descriptors."""

from argparse import ArgumentParser, Namespace


class BaseCommand:
    """Represents the base class for all commands in the application."""

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.

        Raises:
            NotImplementedError: Since this is an abstract base class, NotImplementedError is raised.
        """

        raise NotImplementedError()


class BaseCommandDescriptor:
    """Represents a description of a command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.

        Raises:
            NotImplementedError: Since this is an abstract base class, NotImplementedError is raised.
        """

        raise NotImplementedError()

    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.

        Raises:
            NotImplementedError: Since this is an abstract base class, NotImplementedError is raised.
        """

        raise NotImplementedError()

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Adds the command line arguments to the command line argument parser.

        Args:
            parser (ArgumentParser): The command line argument parser to which the arguments are to be added.

        Raises:
            NotImplementedError: Since this is an abstract base class, NotImplementedError is raised.
        """

        raise NotImplementedError()
