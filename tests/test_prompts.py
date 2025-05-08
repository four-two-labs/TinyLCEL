# Standard library
import asyncio
import time # For abatch concurrency test
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import call # Added for call object

# Third-party
import pytest

# Project-specific
from tinylcel.messages import AIMessage
from tinylcel.messages import BaseMessage
from tinylcel.messages import HumanMessage
from tinylcel.messages import SystemMessage
from tinylcel.prompts import ChatPromptTemplate
from tinylcel.prompts import MessageTemplateType


# --- Fixtures ---

@pytest.fixture
def basic_template_tuples() -> List[Tuple[str, str]]:
    return [
        ('system', 'You are {bot_name}.'),
        ('human', 'Hello, I am {user_name}.'),
        ('ai', 'Nice to meet you, {user_name}!'),
    ]

@pytest.fixture
def basic_template_objects() -> List[BaseMessage]:
    return [
        SystemMessage(content='You are {bot_name}.'),
        HumanMessage(content='Hello, I am {user_name}.'),
        AIMessage(content='Nice to meet you, {user_name}!'),
    ]

@pytest.fixture
def mixed_template_input(basic_template_tuples, basic_template_objects) -> List[MessageTemplateType]:
    return [
        basic_template_tuples[0], # ('system', 'You are {bot_name}.')
        basic_template_objects[1], # HumanMessage(content='Hello, I am {user_name}.')
        ('ai', 'Nice to meet you, {user_name}! I am {bot_name}.'), # Tuple with different content
    ]

@pytest.fixture
def complex_vars_template_tuples() -> List[Tuple[str, str]]:
    # Use dictionary-style access for str.format()
    return [
        ('system', 'User: {user[name]}, Age: {user[age]}. Query about {item[0]}.'),
        ('human', 'The id is {id}.'),
    ]

@pytest.fixture
def sample_input_values() -> Dict[str, Any]:
    return {
        'bot_name': 'Chatty',
        'user_name': 'Alice',
        'user': {'name': 'Bob', 'age': 30}, # user is a dict
        'item': ['productX'], # item is a list
        'id': 123,
    }

# --- Test Cases ---

class TestChatPromptTemplateInitialization:
    def test_init_with_tuples(self, basic_template_tuples):
        prompt = ChatPromptTemplate(basic_template_tuples)
        assert len(prompt.message_templates) == 3
        assert prompt.message_templates[0] == ('system', 'You are {bot_name}.')

    def test_init_with_base_message_objects(self, basic_template_objects):
        prompt = ChatPromptTemplate(basic_template_objects)
        assert len(prompt.message_templates) == 3
        assert prompt.message_templates[0] == ('system', 'You are {bot_name}.')
        assert prompt.message_templates[1] == ('human', 'Hello, I am {user_name}.')

    def test_init_with_mixed_types(self, mixed_template_input):
        prompt = ChatPromptTemplate(mixed_template_input)
        assert len(prompt.message_templates) == 3
        assert prompt.message_templates[0] == ('system', 'You are {bot_name}.')
        assert prompt.message_templates[1] == ('human', 'Hello, I am {user_name}.')
        assert prompt.message_templates[2] == ('ai', 'Nice to meet you, {user_name}! I am {bot_name}.')

    def test_init_invalid_tuple_structure(self):
        with pytest.raises(TypeError, match="must be a \\(role: str, template: str\\) tuple or a BaseMessage instance"):
            ChatPromptTemplate([('system', 'Valid'), ('human', 123, 'Extra')]) # type: ignore

    def test_init_invalid_role_in_tuple(self):
        with pytest.raises(ValueError, match="Item at index 1 \\(tuple\\) has an invalid role: 'unknown_role'"):
            ChatPromptTemplate([('system', 'Valid'), ('unknown_role', 'Template')])

    def test_init_invalid_type_in_list(self):
        with pytest.raises(TypeError, match="Item at index 1 in 'message_templates' must be a \\(role: str, template: str\\) tuple or a BaseMessage instance"):
            ChatPromptTemplate([('system', 'Valid'), 123]) # type: ignore

    def test_from_messages_valid_tuples(self, basic_template_tuples):
        prompt = ChatPromptTemplate.from_messages(basic_template_tuples) # type: ignore
        assert len(prompt.message_templates) == 3
        assert prompt.message_templates[0] == ('system', 'You are {bot_name}.')
    
    # Removed test_from_messages_invalid_item_type_in_list as it's no longer applicable
    # due to from_messages directly calling __init__ which correctly handles mixed types.


class TestChatPromptTemplateInputVariables:
    def test_input_variables_property_simple(self, basic_template_tuples):
        prompt = ChatPromptTemplate(basic_template_tuples)
        assert prompt.input_variables == ['bot_name', 'user_name']

    def test_input_variables_property_complex(self, complex_vars_template_tuples):
        prompt = ChatPromptTemplate(complex_vars_template_tuples)
        # _extract_input_variables should correctly get base names 'user' and 'item'
        assert prompt.input_variables == ['id', 'item', 'user']

    def test_extract_input_variables_empty(self):
        prompt = ChatPromptTemplate([])
        assert prompt.input_variables == []
        assert prompt._extract_input_variables([]) == set()

    def test_extract_input_variables_no_vars(self):
        prompt = ChatPromptTemplate([('system', 'Hello world.')])
        assert prompt.input_variables == []

    def test_input_variables_are_sorted_and_unique(self):
        templates = [('human', '{b} {a} {b} {c.d} {c[name]}')]
        prompt = ChatPromptTemplate(templates)
        assert prompt.input_variables == ['a', 'b', 'c']

    def test_init_empty_placeholder_name(self):
        templates1 = [("human", "Value is { }")] # Formatter parses ' ' as field_name
        prompt1 = ChatPromptTemplate(templates1)
        assert prompt1.input_variables == [], "Space-only placeholder should be ignored"

        templates2 = [("human", "Value is {}")] # Formatter parses '' as field_name
        prompt2 = ChatPromptTemplate(templates2)
        assert prompt2.input_variables == [], "Empty placeholder should be ignored"

        templates3 = [("human", "Value is { [ ]}")] # field_name might be ' [ ]'
        prompt3 = ChatPromptTemplate(templates3)
        assert prompt3.input_variables == [], "Spaced list-like empty placeholder should be ignored"
        
        templates4 = [("human", "Value is {name} and { }")]
        prompt4 = ChatPromptTemplate(templates4)
        assert prompt4.input_variables == ['name'], "Valid var should be present, empty ignored"


class TestChatPromptTemplateFormatting:
    def test_format_messages_success(self, basic_template_tuples, sample_input_values):
        prompt = ChatPromptTemplate(basic_template_tuples)
        formatted = prompt.format_messages(bot_name='TestBot', user_name='TestUser')
        
        assert len(formatted) == 3
        assert isinstance(formatted[0], SystemMessage)
        assert formatted[0].content == 'You are TestBot.'
        assert formatted[0].role == 'system'
        
        assert isinstance(formatted[1], HumanMessage)
        assert formatted[1].content == 'Hello, I am TestUser.'
        assert formatted[1].role == 'human'

        assert isinstance(formatted[2], AIMessage)
        assert formatted[2].content == 'Nice to meet you, TestUser!'
        assert formatted[2].role == 'ai'

    def test_format_messages_with_complex_vars(self, complex_vars_template_tuples, sample_input_values):
        prompt = ChatPromptTemplate(complex_vars_template_tuples)
        formatted = prompt.format_messages(**sample_input_values)
        assert formatted[0].content == "User: Bob, Age: 30. Query about productX."
        assert formatted[1].content == "The id is 123."

    def test_format_messages_missing_input_variable_error(self, basic_template_tuples):
        prompt = ChatPromptTemplate(basic_template_tuples)
        with pytest.raises(KeyError, match=r"Missing input variables: \['bot_name'\]. Provided variables: \['user_name'\]"):
            prompt.format_messages(user_name='TestUser')

    def test_format_messages_template_string_key_error(self):
        templates = [('human', 'Hello {name}, you have {message}. Your code is {code[value]}')]
        prompt = ChatPromptTemplate(templates) # input_variables will be ['code', 'message', 'name']
        
        # Case 1: A top-level variable identified by input_variables is missing from kwargs.
        with pytest.raises(KeyError, match=r"Missing input variables: \['message'\]. Provided variables: \['code', 'name'\]"):
            prompt.format_messages(name='User', code={'value': 1})

        # Case 2: An item access fails during format() inside template_str.format()
        # This will raise a KeyError from within template_str.format(), which is then
        # caught and re-raised by the format_message inner function.
        with pytest.raises(KeyError, match=r"A variable required by the template string was not found in the provided arguments. Missing key: 'value'. Template: 'Hello {name}, you have {message}. Your code is {code\[value\]}'"):
            prompt.format_messages(name='User', message='A message', code={'actual_value_is_missing': 1})


    def test_format_messages_value_error_for_object_attribute_format_issue(self, sample_input_values):
        # This test is for when str.format() tries to access an attribute that doesn't exist on an object.
        # To test this, we need an object, not just a dict for {user.non_existent_attr}
        class DummyUser:
            def __init__(self, name):
                self.name = name
                # non_existent_attr is missing

        templates = [('human', 'User name: {user.name}, User detail: {user.non_existent_attr}')]
        prompt = ChatPromptTemplate(templates)
        
        current_input_values = sample_input_values.copy()
        current_input_values['user'] = DummyUser(name='TestDummy')

        with pytest.raises(ValueError, match="Error formatting template.*AttributeError: .*DummyUser.* object has no attribute 'non_existent_attr'"):
            prompt.format_messages(**current_input_values)

    def test_format_messages_ignores_extra_kwargs(self, basic_template_tuples):
        prompt = ChatPromptTemplate(basic_template_tuples)
        # Provide an extra kwarg not defined in the templates
        formatted = prompt.format_messages(
            bot_name='TestBot', 
            user_name='TestUser',
            extra_variable='should_be_ignored'
        )
        assert len(formatted) == 3
        assert isinstance(formatted[0], SystemMessage)
        assert formatted[0].content == 'You are TestBot.'
        assert formatted[0].role == 'system'
        assert isinstance(formatted[1], HumanMessage)
        assert formatted[1].content == 'Hello, I am TestUser.'
        assert formatted[1].role == 'human'
        # Ensure no error was raised due to extra_variable


class TestChatPromptTemplateRunnableMethods:
    @pytest.fixture
    def prompt(self, basic_template_tuples) -> ChatPromptTemplate:
        return ChatPromptTemplate(basic_template_tuples)

    @pytest.fixture
    def valid_invoke_input(self) -> Dict[str, Any]:
        return {'bot_name': 'Invoker', 'user_name': 'Test'}

    def test_invoke_success(self, prompt, valid_invoke_input):
        result = prompt.invoke(valid_invoke_input)
        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == 'You are Invoker.'

    def test_invoke_error_missing_var(self, prompt):
        with pytest.raises(KeyError, match="Missing input variables: \\['user_name'\\]"):
            prompt.invoke({'bot_name': 'ErrorBot'})

    @pytest.mark.asyncio
    async def test_ainvoke_success(self, prompt, valid_invoke_input):
        result = await prompt.ainvoke(valid_invoke_input)
        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == 'You are Invoker.'

    @pytest.mark.asyncio
    async def test_ainvoke_error_missing_var(self, prompt):
        with pytest.raises(KeyError, match="Missing input variables: \\['user_name'\\]"):
            await prompt.ainvoke({'bot_name': 'AsyncErrorBot'})

    def test_batch_success(self, prompt, valid_invoke_input):
        inputs = [valid_invoke_input, {'bot_name': 'BatchBot', 'user_name': 'BatchUser'}]
        results = prompt.batch(inputs)
        assert len(results) == 2
        assert results[0][0].content == 'You are Invoker.'
        assert results[1][1].content == 'Hello, I am BatchUser.'

    def test_batch_return_exceptions(self, prompt, valid_invoke_input):
        inputs = [
            valid_invoke_input, 
            {'bot_name': 'ErrorBot'}, # Missing user_name
            {'bot_name': 'GoodBot', 'user_name': 'GoodUser'}
        ]
        results = prompt.batch(inputs, return_exceptions=True)
        assert len(results) == 3
        assert isinstance(results[0], list) # Successful result is a list of messages
        assert isinstance(results[0][0], SystemMessage)
        assert isinstance(results[1], KeyError) # Exception for the second input
        assert "Missing input variables: ['user_name']" in str(results[1])
        assert isinstance(results[2], list)
        assert results[2][0].content == "You are GoodBot."


    @pytest.mark.asyncio
    async def test_abatch_success(self, prompt, valid_invoke_input):
        inputs = [valid_invoke_input, {'bot_name': 'AsyncBatchBot', 'user_name': 'AsyncBatchUser'}]
        results = await prompt.abatch(inputs)
        assert len(results) == 2
        assert results[0][0].content == 'You are Invoker.'
        assert results[1][1].content == 'Hello, I am AsyncBatchUser.'

    @pytest.mark.asyncio
    async def test_abatch_return_exceptions(self, prompt, valid_invoke_input):
        inputs = [
            valid_invoke_input,
            {'bot_name': 'AsyncErrorBot'}, # Missing user_name
            {'bot_name': 'AsyncGoodBot', 'user_name': 'AsyncGoodUser'}
        ]
        results = await prompt.abatch(inputs, return_exceptions=True)
        assert len(results) == 3
        assert isinstance(results[0], list)
        assert isinstance(results[1], KeyError)
        assert "Missing input variables: ['user_name']" in str(results[1])
        assert isinstance(results[2], list)
        assert results[2][0].content == "You are AsyncGoodBot."

    @pytest.mark.asyncio
    async def test_abatch_concurrency(self, basic_template_tuples):
        original_format_messages = ChatPromptTemplate.format_messages
        call_log = []
        
        prompt_instance = ChatPromptTemplate(basic_template_tuples)
        
        # To simulate async work within format_messages for the purpose of this test,
        # we need to mock the ainvoke method of the instance to call a version
        # of format_messages that includes an await asyncio.sleep().
        # This is because the default RunnableBase.abatch calls ainvoke, and the
        # default ChatPromptTemplate.ainvoke calls the synchronous format_messages.

        async def mock_ainvoke_with_delay(input_dict):
            call_log.append(f"start: {input_dict.get('user_name', 'unknown')}")
            await asyncio.sleep(0.02) # Simulate async work
            # Call the original synchronous format_messages logic
            result = original_format_messages(prompt_instance, **input_dict)
            call_log.append(f"end: {input_dict.get('user_name', 'unknown')}")
            return result

        prompt_instance.ainvoke = AsyncMock(side_effect=mock_ainvoke_with_delay)

        inputs = [
            {'bot_name': 'B1', 'user_name': 'U1'},
            {'bot_name': 'B2', 'user_name': 'U2'},
            {'bot_name': 'B3', 'user_name': 'U3'},
            {'bot_name': 'B4', 'user_name': 'U4'},
        ]
        max_concurrency = 2
        
        start_time = time.monotonic()
        results = await prompt_instance.abatch(inputs, max_concurrency=max_concurrency)
        end_time = time.monotonic()

        assert len(results) == 4
        assert len(call_log) == 8 # 4 starts, 4 ends
        prompt_instance.ainvoke.assert_has_awaits([
            call({'bot_name': 'B1', 'user_name': 'U1'}),
            call({'bot_name': 'B2', 'user_name': 'U2'}),
            call({'bot_name': 'B3', 'user_name': 'U3'}),
            call({'bot_name': 'B4', 'user_name': 'U4'}),
        ], any_order=True) # Check that ainvoke was called for all inputs

        expected_min_time = 2 * 0.02 
        expected_max_time = (2 * 0.02) + 0.03 
        actual_time = end_time - start_time
        
        assert actual_time >= expected_min_time, f"Expected min time {expected_min_time:.3f}s, got {actual_time:.3f}s"
        assert actual_time < expected_max_time, f"Expected max time {expected_max_time:.3f}s, got {actual_time:.3f}s. Log: {call_log}"
        
        assert (call_log[0].startswith("start: U1") and call_log[1].startswith("start: U2")) or \
               (call_log[0].startswith("start: U2") and call_log[1].startswith("start: U1")) 