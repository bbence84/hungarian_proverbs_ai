from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel import Kernel
from dotenv import dotenv_values, find_dotenv
from typing import Annotated

class UtilityPlugin(KernelBaseModel):

    @classmethod
    async def create(cls, kernel: Kernel):
        self = cls(kernel)
        return self    

    def __init__(self, kernel: Kernel):
        super().__init__()
        self._kernel = kernel

    @kernel_function( description="Create an HTML file with the specified content and file name.", 
                    name="CreateHTMLFile", )
    def create_html_file(self, html_content: str, file_name: str):
        """
        Create an HTML file with the specified content and file name.

        :param html_content: The content of the HTML file
        :param file_name: The name of the HTML file
        """
        with open(file_name, "w") as file:
            file.write(html_content)