# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is
# regenerated.
# --------------------------------------------------------------------------

from msrest.serialization import Model


class ContentHash(Model):
    """Definition of the runbook property type.

    All required parameters must be populated in order to send to Azure.

    :param algorithm: Required. Gets or sets the content hash algorithm used
     to hash the content.
    :type algorithm: str
    :param value: Required. Gets or sets expected hash value of the content.
    :type value: str
    """

    _validation = {
        'algorithm': {'required': True},
        'value': {'required': True},
    }

    _attribute_map = {
        'algorithm': {'key': 'algorithm', 'type': 'str'},
        'value': {'key': 'value', 'type': 'str'},
    }

    def __init__(self, *, algorithm: str, value: str, **kwargs) -> None:
        super(ContentHash, self).__init__(**kwargs)
        self.algorithm = algorithm
        self.value = value