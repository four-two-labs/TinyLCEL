import typing as t
from dataclasses import field
from dataclasses import dataclass

from pydantic import BaseModel

from tinylcel.output_parsers import BaseOutputParser
from tinylcel.output_parsers import JsonOutputParser


@dataclass(frozen=True)
class PydanticOutputParser[T: BaseModel](BaseOutputParser[T]):
    """Output parser for Pydantic models that validates JSON output against a schema.

    This parser extends JsonOutputParser to add Pydantic model validation.
    It first parses the input as JSON using the parent class, then validates
    the parsed data against the specified Pydantic model schema.

    Type Parameters:
        T: A Pydantic BaseModel class that defines the expected output schema.

    Attributes:
        model: The Pydantic model class to validate against.

    Example:
        ```python
        from pydantic import BaseModel
        from tinylcel.output_parsers import PydanticOutputParser


        class Person(BaseModel):
            name: str
            age: int


        parser = PydanticOutputParser(model=Person)
        result = parser.parse(message_with_json_content)
        # result is now a Person instance with validated data
        ```

    Raises:
        ParseError: If the input cannot be parsed as JSON (inherited from JsonOutputParser).
        ValidationError: If the parsed JSON doesn't match the Pydantic model schema.
    """

    model: type[T]
    parser: JsonOutputParser = field(default_factory=JsonOutputParser)

    def parse(self, input: t.Any) -> T:  # type: ignore[override]
        """Parse the JSON output into a validated Pydantic model instance.

        This method first delegates to the parent JsonOutputParser to extract
        and parse JSON content, then validates the result against the specified
        Pydantic model schema.

        Args:
            input: An object with a 'content' attribute containing JSON data
                (typically an AIMessage or similar message object).

        Returns:
            A validated instance of the Pydantic model with the parsed data.

        Raises:
            ParseError: If the input cannot be parsed as JSON or extracted from
                the input object (inherited from parent classes).
            ValidationError: If the parsed JSON data doesn't conform to the
                Pydantic model schema (e.g., missing required fields, wrong types).

        Example:
            ```python
            from pydantic import BaseModel
            from tinylcel.messages import AIMessage


            class User(BaseModel):
                username: str
                email: str


            parser = PydanticOutputParser(model=User)
            message = AIMessage(content='{"username": "john", "email": "john@example.com"}')
            user = parser.parse(message)  # Returns User instance
            ```
        """
        parsed = self.parser.parse(input)
        return self.model.model_validate(parsed)
