import string
from typing import Any
from typing import ClassVar

from tinylcel.messages import AIMessage
from tinylcel.messages import BaseMessage
from tinylcel.messages import HumanMessage
from tinylcel.runnable import RunnableBase
from tinylcel.messages import SystemMessage

type MessageTemplateType = tuple[str, str] | BaseMessage


class ChatPromptTemplate(RunnableBase[dict[str, Any], list[BaseMessage]]):
    """A template for chat models.

    Equivalent to a simplified version of Langchain's ChatPromptTemplate,
    focusing on formatting messages from templates.
    Outputs a list of BaseMessage instances (HumanMessage, AIMessage, SystemMessage).
    """

    VALID_ROLES: ClassVar[set[str]] = {'human', 'ai', 'system'}

    def __init__(self, message_templates: list[MessageTemplateType]):
        """
        Initializes the ChatPromptTemplate.

        Args:
            message_templates: A list of message template inputs. Each item can be:
                - A tuple of (role: str, template_string: str).
                  Example: ('system', 'You are {name}.')
                - A BaseMessage instance (HumanMessage, AIMessage, SystemMessage).
                  Example: SystemMessage(content='You are {name}.')

        Raises:
            TypeError: If an item in `message_templates` is not a valid type.
            ValueError: If a tuple-based message has an invalid structure or role.
        """
        processed_templates: list[tuple[str, str]] = []
        for i, item in enumerate(message_templates):
            match item:
                case (str() as role, str() as template_str):
                    if role not in self.VALID_ROLES:
                        raise ValueError(
                            f"Item at index {i} (tuple) has an invalid role: '{role}'. "
                            f'Valid roles are: {self.VALID_ROLES}.'
                        )
                    processed_templates.append((role, template_str))
                case BaseMessage() as msg_obj:
                    if not isinstance(msg_obj.content, str):
                        raise TypeError(
                            f'BaseMessage content must be a string for template use, got {type(msg_obj.content)} '
                            f'at index {i}. Multimodal content is not supported in templates.'
                        )
                    processed_templates.append((msg_obj.role, msg_obj.content))
                case _:
                    raise TypeError(
                        f"Item at index {i} in 'message_templates' must be a (role: str, template: str) tuple "
                        f'or a BaseMessage instance. Got: {type(item)}'
                    )

        self.message_templates: list[tuple[str, str]] = processed_templates
        self._input_variables_set: set[str] = self._extract_input_variables(self.message_templates)

    def _extract_input_variables(self, message_templates: list[tuple[str, str]]) -> set[str]:
        """Extracts all unique base input variable names from the template strings.

        For a template like "{user.name}, your query is {query}", it extracts {'user', 'query'}.
        """
        variables: set[str] = set()
        formatter = string.Formatter()
        for _, template_str in message_templates:
            for _, field_name, _, _ in formatter.parse(template_str):
                if field_name is not None and field_name != '':
                    # Extracts the base variable name. E.g., 'person.name' -> 'person', 'data[0]' -> 'data'
                    base_variable_name = field_name.split('.')[0].split('[')[0].strip()
                    if base_variable_name:  # Ensure it's not an empty string after splits
                        variables.add(base_variable_name)
        return variables

    @property
    def input_variables(self) -> list[str]:
        """Returns a sorted list of unique input variable names expected by the template.

        These are the keys required in the input dictionary for formatting.
        """
        return sorted(self._input_variables_set)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """
        Formats the message templates with the provided keyword arguments.

        Args:
            **kwargs: The variables to substitute into the templates.
                      All keys from `input_variables` must be present.

        Returns:
            A list of formatted messages (HumanMessage, AIMessage, SystemMessage instances).

        Raises:
            KeyError: If an input variable identified in `input_variables`
                      is not provided in `kwargs`.
            ValueError: If an error occurs during string formatting or if an
                        unknown role is encountered.
        """

        def format_message(role_str: str, template_str: str) -> BaseMessage:
            try:
                content = template_str.format(**kwargs)
            except KeyError as e:
                raise KeyError(
                    'A variable required by the template string was not found in the '
                    f"provided arguments. Missing key: '{e.args[0]}'. "
                    f"Template: '{template_str}'"
                ) from e
            except Exception as e:
                raise ValueError(
                    f"Error formatting template: '{template_str}'. "
                    f'Input arguments: {list(kwargs.keys())}. Error: {type(e).__name__}: {e}'
                ) from e

            match role_str:
                case 'human':
                    return HumanMessage(content=content)
                case 'ai':
                    return AIMessage(content=content)
                case 'system':
                    return SystemMessage(content=content)

            raise ValueError(f"Unknown role: '{role_str}'. Must be 'human', 'ai', or 'system'.")

        missing_vars = self._input_variables_set - kwargs.keys()
        if missing_vars:
            raise KeyError(
                f'Missing input variables: {sorted(missing_vars)}. Provided variables: {sorted(kwargs.keys())}'
            )

        return [format_message(role_str, template_str) for role_str, template_str in self.message_templates]

    def invoke(self, input: dict[str, Any]) -> list[BaseMessage]:
        """
        Formats the prompt with the given input.

        Args:
            input: A dictionary of key-value pairs to fill into the prompt.
                   All keys from `input_variables` must be present.

        Returns:
            A list of formatted BaseMessage instances.

        Raises:
            KeyError: If a required variable is missing from the input.
            ValueError: If an error occurs during string formatting or role assignment.
        """
        return self.format_messages(**input)

    async def ainvoke(self, input: dict[str, Any]) -> list[BaseMessage]:
        """
        Asynchronously formats the prompt with the given input.

        Args:
            input: A dictionary of key-value pairs to fill into the prompt.
                   All keys from `input_variables` must be present.

        Returns:
            A list of formatted BaseMessage instances.

        Raises:
            KeyError: If a required variable is missing from the input.
            ValueError: If an error occurs during string formatting or role assignment.
        """
        return self.format_messages(**input)

    @classmethod
    def from_messages(cls, messages: list[MessageTemplateType]) -> 'ChatPromptTemplate':
        """
        Creates a ChatPromptTemplate from a list of (role, template_string) tuples.

        Args:
            messages: A list of tuples, where each tuple must contain
                      (role: str, template_string: str). Roles should be
                      'human', 'ai', or 'system'.
                      Example: [('system', 'You are {name}.'),
                                ('human', 'Hi {name}, {question}')]

        Returns:
            An instance of ChatPromptTemplate.

        Raises:
            TypeError: If `messages` is not a list.
            ValueError: If any item in `messages` is not a 2-tuple of strings.
                        (Role string validity is checked by __init__).
        """
        return cls(messages)
