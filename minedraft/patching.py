# This file is modified from https://github.com/snowflakedb/ArcticInference/blob/3461c38e63a95f88e6e6f61c9c521760486dc344/arctic_inference/patching.py

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from types import MethodType, ModuleType
from typing import Callable, Type, Union

from vllm.logger import init_logger

logger = init_logger(__name__)

Patchable = Union[Type, ModuleType]


def is_module_function(target: Patchable, name: str) -> bool:
    """
    Check if getattr(target, name) is a module-level function.
    
    Args:
        target: A class or module
        name: The name of the attribute to check
    
    Returns:
        bool: True if the attribute is a module-level function, False otherwise.
    """
    # If target is not a module, it cannot have module-level functions
    if not inspect.ismodule(target):
        return False

    attr = getattr(target, name, None)

    # If the attribute is not found or is not a function, return False
    if attr is None or not inspect.isfunction(attr):
        return False

    # Check if the attribute is defined in the module's namespace:
    # - `attr.__module__` should be equal to the module's name
    # - `attr.__qualname__` should be equal to the name of the function
    return attr.__module__ == target.__name__ and attr.__qualname__ == name

def patch_module_function(target: ModuleType, name: str, replacement: Callable):
    """
    Patch the module function by replacing its internal bytecodes, defaults,
    annotations, and globals with those from the replacement function.
    
    Args:
        target: The target module.
        name: The name of the module function to patch.
        replacement: The replacement function.
    """
    # Ensure that the unwrapped replacement function does not have a closure
    replacement = inspect.unwrap(replacement)
    assert getattr(replacement, '__closure__', None) is None, \
        f"Patch {replacement.__name__} cannot have a closure"

    # Ensure that the unwrapped target function does not have a closure
    target_attr = inspect.unwrap(getattr(target, name))
    assert getattr(target_attr, '__closure__', None) is None, \
        f"Cannot patch module-level function {name} with a closure"

    # Replace the target function's internal objects with those from the replacement
    target_attr.__code__ = replacement.__code__
    target_attr.__defaults__ = replacement.__defaults__
    target_attr.__kwdefaults__ = replacement.__kwdefaults__
    target_attr.__annotations__ = replacement.__annotations__
    if getattr(replacement, '__globals__', None) is not None:
        for key in replacement.__code__.co_names:
            if key in replacement.__globals__:
                target_attr.__globals__[key] = replacement.__globals__[key]


class MinePatch:
    """
    MinePatch provides a mechanism for cleanly patching (extending or
    modifying) existing classes or modules.

    This class uses a subscription syntax to specify the target class or
    module to be patched. Subclasses of MinePatch should define new or
    replacement attributes and methods that will be applied in-place to the
    target when `apply_patch()` is called.

    NOTE: Original module functions being patched are *inaccessible* after
    patching.

    Example 1: Patching a class

    ```python
    # Define a class patch with new methods
    class ExamplePatch(MinePatch[SomeClass]):

        new_field = "This field will be added to SomeClass"

        def new_method(self):
            return "This method will be added to SomeClass"

        @classmethod
        def new_classmethod(cls):
            return "This classmethod will be added to SomeClass"

    # Apply the patch to the target class
    ExamplePatch.apply_patch()

    # Now these methods are available on the original class
    instance = SomeClass()
    instance.new_method()  # Works!
    SomeClass.new_class_method()  # Works!
    ```

    Example 2: Patching a module

    ```python
    # Define a module patch
    class ModulePatch(MinePatch[some_module]):
        NEW_CONSTANT = "This will be added to some_module"

        @staticmethod
        def new_function():
            return "This function will be added to some_module"

    ModulePatch.apply_patch()

    # The constant and function are now available in the module
    some_module.NEW_CONSTANT  # Works!
    some_module.new_function()  # Works!
    ```
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure that subclasses are created using the subscript syntax.
        if not hasattr(cls, '_mine_patch_target'):
            raise TypeError("Subclasses of MinePatch must be defined as "
                            "MinePatch[Target] to specify a patch target")

    @classmethod
    def __class_getitem__(cls, target: Patchable) -> Type:
        # The dynamic type created here will carry the target class as
        # _mine_patch_target.
        if not isinstance(target, (type, ModuleType)):
            raise TypeError(f"MinePatch can only target a class or module, "
                            f"not {type(target)}")
        return type(f"{cls.__name__}[{target.__name__}]", (cls,),
                    {'_mine_patch_target': target})

    @classmethod
    def apply_patch(cls) -> None:
        """
        Patches the target class or module by replacing its attributes with
        those defined on the MinePatch subclass. Attributes are directly
        assigned to the target, and classmethods are re-bound to the target
        class before assignment.

        Raises:
            TypeError: If the MinePatch subclass is not defined with a target
                class or module.
        """
        if cls is MinePatch or not issubclass(cls, MinePatch):
            raise TypeError("apply_patch() must be called on a subclass of "
                            "MinePatch")

        target = cls._mine_patch_target

        if "_mine_patches" not in target.__dict__:
            target._mine_patches = {}

        for name, replacement in cls.__dict__.items():

            # Skip special names and the '_mine_patch_target' itself
            if name in ("_mine_patch_target", "__dict__", "__weakref__",
                        "__module__", "__doc__", "__parameters__",):
                continue

            # Warn if the attribute has already been patched
            if name in target._mine_patches:
                patch = target._mine_patches[name]
                logger.warning(f"{target.__name__}.{name} is already "
                               f"patched by {patch.__name__}")
            target._mine_patches[name] = cls

            # If module function, update the respective function's
            # internal bytecodes, defaults, annotations, and globals.
            if is_module_function(target, name):
                assert isinstance(replacement, staticmethod), \
                    f"{cls.__name__} cannot patch module-level function {name} " \
                    f"with a non-static method"

                replacement = replacement.__wrapped__

                patch_module_function(target, name, replacement)
                logger.info(f"{cls.__name__} replaced {target.__name__}.{name}")
                continue

            # If classmethod, re-bind it to the target
            if isinstance(replacement, MethodType):
                replacement = MethodType(replacement.__func__, target)

            # Patch the target with the new attribute
            replace = hasattr(target, name)
            setattr(target, name, replacement)
            action = "replaced" if replace else "added"
            logger.info(f"{cls.__name__} {action} {target.__name__}.{name}")
