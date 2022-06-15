# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

import datetime
from typing import Dict, List, Optional, Union

from azure.core.exceptions import HttpResponseError
import msrest.serialization

from ._nginx_management_client_enums import *


class ErrorResponseBody(msrest.serialization.Model):
    """ErrorResponseBody.

    :ivar code:
    :vartype code: str
    :ivar message:
    :vartype message: str
    :ivar target:
    :vartype target: str
    :ivar details:
    :vartype details: list[~azure.mgmt.nginx.models.ErrorResponseBody]
    """

    _attribute_map = {
        'code': {'key': 'code', 'type': 'str'},
        'message': {'key': 'message', 'type': 'str'},
        'target': {'key': 'target', 'type': 'str'},
        'details': {'key': 'details', 'type': '[ErrorResponseBody]'},
    }

    def __init__(
        self,
        *,
        code: Optional[str] = None,
        message: Optional[str] = None,
        target: Optional[str] = None,
        details: Optional[List["ErrorResponseBody"]] = None,
        **kwargs
    ):
        """
        :keyword code:
        :paramtype code: str
        :keyword message:
        :paramtype message: str
        :keyword target:
        :paramtype target: str
        :keyword details:
        :paramtype details: list[~azure.mgmt.nginx.models.ErrorResponseBody]
        """
        super(ErrorResponseBody, self).__init__(**kwargs)
        self.code = code
        self.message = message
        self.target = target
        self.details = details


class IdentityProperties(msrest.serialization.Model):
    """IdentityProperties.

    Variables are only populated by the server, and will be ignored when sending a request.

    :ivar principal_id:
    :vartype principal_id: str
    :ivar tenant_id:
    :vartype tenant_id: str
    :ivar type: Possible values include: "SystemAssigned", "UserAssigned", "SystemAssigned,
     UserAssigned", "None".
    :vartype type: str or ~azure.mgmt.nginx.models.IdentityType
    :ivar user_assigned_identities: Dictionary of :code:`<UserIdentityProperties>`.
    :vartype user_assigned_identities: dict[str, ~azure.mgmt.nginx.models.UserIdentityProperties]
    """

    _validation = {
        'principal_id': {'readonly': True},
        'tenant_id': {'readonly': True},
    }

    _attribute_map = {
        'principal_id': {'key': 'principalId', 'type': 'str'},
        'tenant_id': {'key': 'tenantId', 'type': 'str'},
        'type': {'key': 'type', 'type': 'str'},
        'user_assigned_identities': {'key': 'userAssignedIdentities', 'type': '{UserIdentityProperties}'},
    }

    def __init__(
        self,
        *,
        type: Optional[Union[str, "IdentityType"]] = None,
        user_assigned_identities: Optional[Dict[str, "UserIdentityProperties"]] = None,
        **kwargs
    ):
        """
        :keyword type: Possible values include: "SystemAssigned", "UserAssigned", "SystemAssigned,
         UserAssigned", "None".
        :paramtype type: str or ~azure.mgmt.nginx.models.IdentityType
        :keyword user_assigned_identities: Dictionary of :code:`<UserIdentityProperties>`.
        :paramtype user_assigned_identities: dict[str, ~azure.mgmt.nginx.models.UserIdentityProperties]
        """
        super(IdentityProperties, self).__init__(**kwargs)
        self.principal_id = None
        self.tenant_id = None
        self.type = type
        self.user_assigned_identities = user_assigned_identities


class NginxCertificate(msrest.serialization.Model):
    """NginxCertificate.

    Variables are only populated by the server, and will be ignored when sending a request.

    :ivar id:
    :vartype id: str
    :ivar name:
    :vartype name: str
    :ivar type:
    :vartype type: str
    :ivar properties:
    :vartype properties: ~azure.mgmt.nginx.models.NginxCertificateProperties
    :ivar tags: A set of tags. Dictionary of :code:`<string>`.
    :vartype tags: dict[str, str]
    :ivar location:
    :vartype location: str
    :ivar system_data: Metadata pertaining to creation and last modification of the resource.
    :vartype system_data: ~azure.mgmt.nginx.models.SystemData
    """

    _validation = {
        'id': {'readonly': True},
        'name': {'readonly': True},
        'type': {'readonly': True},
        'system_data': {'readonly': True},
    }

    _attribute_map = {
        'id': {'key': 'id', 'type': 'str'},
        'name': {'key': 'name', 'type': 'str'},
        'type': {'key': 'type', 'type': 'str'},
        'properties': {'key': 'properties', 'type': 'NginxCertificateProperties'},
        'tags': {'key': 'tags', 'type': '{str}'},
        'location': {'key': 'location', 'type': 'str'},
        'system_data': {'key': 'systemData', 'type': 'SystemData'},
    }

    def __init__(
        self,
        *,
        properties: Optional["NginxCertificateProperties"] = None,
        tags: Optional[Dict[str, str]] = None,
        location: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword properties:
        :paramtype properties: ~azure.mgmt.nginx.models.NginxCertificateProperties
        :keyword tags: A set of tags. Dictionary of :code:`<string>`.
        :paramtype tags: dict[str, str]
        :keyword location:
        :paramtype location: str
        """
        super(NginxCertificate, self).__init__(**kwargs)
        self.id = None
        self.name = None
        self.type = None
        self.properties = properties
        self.tags = tags
        self.location = location
        self.system_data = None


class NginxCertificateListResponse(msrest.serialization.Model):
    """NginxCertificateListResponse.

    :ivar value:
    :vartype value: list[~azure.mgmt.nginx.models.NginxCertificate]
    :ivar next_link:
    :vartype next_link: str
    """

    _attribute_map = {
        'value': {'key': 'value', 'type': '[NginxCertificate]'},
        'next_link': {'key': 'nextLink', 'type': 'str'},
    }

    def __init__(
        self,
        *,
        value: Optional[List["NginxCertificate"]] = None,
        next_link: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword value:
        :paramtype value: list[~azure.mgmt.nginx.models.NginxCertificate]
        :keyword next_link:
        :paramtype next_link: str
        """
        super(NginxCertificateListResponse, self).__init__(**kwargs)
        self.value = value
        self.next_link = next_link


class NginxCertificateProperties(msrest.serialization.Model):
    """NginxCertificateProperties.

    :ivar provisioning_state: Possible values include: "Accepted", "Creating", "Updating",
     "Deleting", "Succeeded", "Failed", "Canceled", "Deleted", "NotSpecified".
    :vartype provisioning_state: str or ~azure.mgmt.nginx.models.ProvisioningState
    :ivar key_virtual_path:
    :vartype key_virtual_path: str
    :ivar certificate_virtual_path:
    :vartype certificate_virtual_path: str
    :ivar key_vault_secret_id:
    :vartype key_vault_secret_id: str
    """

    _attribute_map = {
        'provisioning_state': {'key': 'provisioningState', 'type': 'str'},
        'key_virtual_path': {'key': 'keyVirtualPath', 'type': 'str'},
        'certificate_virtual_path': {'key': 'certificateVirtualPath', 'type': 'str'},
        'key_vault_secret_id': {'key': 'keyVaultSecretId', 'type': 'str'},
    }

    def __init__(
        self,
        *,
        provisioning_state: Optional[Union[str, "ProvisioningState"]] = None,
        key_virtual_path: Optional[str] = None,
        certificate_virtual_path: Optional[str] = None,
        key_vault_secret_id: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword provisioning_state: Possible values include: "Accepted", "Creating", "Updating",
         "Deleting", "Succeeded", "Failed", "Canceled", "Deleted", "NotSpecified".
        :paramtype provisioning_state: str or ~azure.mgmt.nginx.models.ProvisioningState
        :keyword key_virtual_path:
        :paramtype key_virtual_path: str
        :keyword certificate_virtual_path:
        :paramtype certificate_virtual_path: str
        :keyword key_vault_secret_id:
        :paramtype key_vault_secret_id: str
        """
        super(NginxCertificateProperties, self).__init__(**kwargs)
        self.provisioning_state = provisioning_state
        self.key_virtual_path = key_virtual_path
        self.certificate_virtual_path = certificate_virtual_path
        self.key_vault_secret_id = key_vault_secret_id


class NginxConfiguration(msrest.serialization.Model):
    """NginxConfiguration.

    Variables are only populated by the server, and will be ignored when sending a request.

    :ivar id:
    :vartype id: str
    :ivar name:
    :vartype name: str
    :ivar type:
    :vartype type: str
    :ivar properties:
    :vartype properties: ~azure.mgmt.nginx.models.NginxConfigurationProperties
    :ivar tags: A set of tags. Dictionary of :code:`<string>`.
    :vartype tags: dict[str, str]
    :ivar location:
    :vartype location: str
    :ivar system_data: Metadata pertaining to creation and last modification of the resource.
    :vartype system_data: ~azure.mgmt.nginx.models.SystemData
    """

    _validation = {
        'id': {'readonly': True},
        'name': {'readonly': True},
        'type': {'readonly': True},
        'system_data': {'readonly': True},
    }

    _attribute_map = {
        'id': {'key': 'id', 'type': 'str'},
        'name': {'key': 'name', 'type': 'str'},
        'type': {'key': 'type', 'type': 'str'},
        'properties': {'key': 'properties', 'type': 'NginxConfigurationProperties'},
        'tags': {'key': 'tags', 'type': '{str}'},
        'location': {'key': 'location', 'type': 'str'},
        'system_data': {'key': 'systemData', 'type': 'SystemData'},
    }

    def __init__(
        self,
        *,
        properties: Optional["NginxConfigurationProperties"] = None,
        tags: Optional[Dict[str, str]] = None,
        location: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword properties:
        :paramtype properties: ~azure.mgmt.nginx.models.NginxConfigurationProperties
        :keyword tags: A set of tags. Dictionary of :code:`<string>`.
        :paramtype tags: dict[str, str]
        :keyword location:
        :paramtype location: str
        """
        super(NginxConfiguration, self).__init__(**kwargs)
        self.id = None
        self.name = None
        self.type = None
        self.properties = properties
        self.tags = tags
        self.location = location
        self.system_data = None


class NginxConfigurationFile(msrest.serialization.Model):
    """NginxConfigurationFile.

    :ivar content:
    :vartype content: str
    :ivar virtual_path:
    :vartype virtual_path: str
    """

    _attribute_map = {
        'content': {'key': 'content', 'type': 'str'},
        'virtual_path': {'key': 'virtualPath', 'type': 'str'},
    }

    def __init__(
        self,
        *,
        content: Optional[str] = None,
        virtual_path: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword content:
        :paramtype content: str
        :keyword virtual_path:
        :paramtype virtual_path: str
        """
        super(NginxConfigurationFile, self).__init__(**kwargs)
        self.content = content
        self.virtual_path = virtual_path


class NginxConfigurationListResponse(msrest.serialization.Model):
    """Response of a list operation.

    :ivar value: Results of a list operation.
    :vartype value: list[~azure.mgmt.nginx.models.NginxConfiguration]
    :ivar next_link: Link to the next set of results, if any.
    :vartype next_link: str
    """

    _attribute_map = {
        'value': {'key': 'value', 'type': '[NginxConfiguration]'},
        'next_link': {'key': 'nextLink', 'type': 'str'},
    }

    def __init__(
        self,
        *,
        value: Optional[List["NginxConfiguration"]] = None,
        next_link: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword value: Results of a list operation.
        :paramtype value: list[~azure.mgmt.nginx.models.NginxConfiguration]
        :keyword next_link: Link to the next set of results, if any.
        :paramtype next_link: str
        """
        super(NginxConfigurationListResponse, self).__init__(**kwargs)
        self.value = value
        self.next_link = next_link


class NginxConfigurationPackage(msrest.serialization.Model):
    """NginxConfigurationPackage.

    :ivar data:
    :vartype data: str
    """

    _attribute_map = {
        'data': {'key': 'data', 'type': 'str'},
    }

    def __init__(
        self,
        *,
        data: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword data:
        :paramtype data: str
        """
        super(NginxConfigurationPackage, self).__init__(**kwargs)
        self.data = data


class NginxConfigurationProperties(msrest.serialization.Model):
    """NginxConfigurationProperties.

    :ivar provisioning_state: Possible values include: "Accepted", "Creating", "Updating",
     "Deleting", "Succeeded", "Failed", "Canceled", "Deleted", "NotSpecified".
    :vartype provisioning_state: str or ~azure.mgmt.nginx.models.ProvisioningState
    :ivar files:
    :vartype files: list[~azure.mgmt.nginx.models.NginxConfigurationFile]
    :ivar package:
    :vartype package: ~azure.mgmt.nginx.models.NginxConfigurationPackage
    :ivar root_file:
    :vartype root_file: str
    """

    _attribute_map = {
        'provisioning_state': {'key': 'provisioningState', 'type': 'str'},
        'files': {'key': 'files', 'type': '[NginxConfigurationFile]'},
        'package': {'key': 'package', 'type': 'NginxConfigurationPackage'},
        'root_file': {'key': 'rootFile', 'type': 'str'},
    }

    def __init__(
        self,
        *,
        provisioning_state: Optional[Union[str, "ProvisioningState"]] = None,
        files: Optional[List["NginxConfigurationFile"]] = None,
        package: Optional["NginxConfigurationPackage"] = None,
        root_file: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword provisioning_state: Possible values include: "Accepted", "Creating", "Updating",
         "Deleting", "Succeeded", "Failed", "Canceled", "Deleted", "NotSpecified".
        :paramtype provisioning_state: str or ~azure.mgmt.nginx.models.ProvisioningState
        :keyword files:
        :paramtype files: list[~azure.mgmt.nginx.models.NginxConfigurationFile]
        :keyword package:
        :paramtype package: ~azure.mgmt.nginx.models.NginxConfigurationPackage
        :keyword root_file:
        :paramtype root_file: str
        """
        super(NginxConfigurationProperties, self).__init__(**kwargs)
        self.provisioning_state = provisioning_state
        self.files = files
        self.package = package
        self.root_file = root_file


class NginxDeployment(msrest.serialization.Model):
    """NginxDeployment.

    Variables are only populated by the server, and will be ignored when sending a request.

    :ivar id:
    :vartype id: str
    :ivar name:
    :vartype name: str
    :ivar type:
    :vartype type: str
    :ivar identity:
    :vartype identity: ~azure.mgmt.nginx.models.IdentityProperties
    :ivar properties:
    :vartype properties: ~azure.mgmt.nginx.models.NginxDeploymentProperties
    :ivar tags: A set of tags. Dictionary of :code:`<string>`.
    :vartype tags: dict[str, str]
    :ivar sku:
    :vartype sku: ~azure.mgmt.nginx.models.ResourceSku
    :ivar location:
    :vartype location: str
    :ivar system_data: Metadata pertaining to creation and last modification of the resource.
    :vartype system_data: ~azure.mgmt.nginx.models.SystemData
    """

    _validation = {
        'id': {'readonly': True},
        'name': {'readonly': True},
        'type': {'readonly': True},
        'system_data': {'readonly': True},
    }

    _attribute_map = {
        'id': {'key': 'id', 'type': 'str'},
        'name': {'key': 'name', 'type': 'str'},
        'type': {'key': 'type', 'type': 'str'},
        'identity': {'key': 'identity', 'type': 'IdentityProperties'},
        'properties': {'key': 'properties', 'type': 'NginxDeploymentProperties'},
        'tags': {'key': 'tags', 'type': '{str}'},
        'sku': {'key': 'sku', 'type': 'ResourceSku'},
        'location': {'key': 'location', 'type': 'str'},
        'system_data': {'key': 'systemData', 'type': 'SystemData'},
    }

    def __init__(
        self,
        *,
        identity: Optional["IdentityProperties"] = None,
        properties: Optional["NginxDeploymentProperties"] = None,
        tags: Optional[Dict[str, str]] = None,
        sku: Optional["ResourceSku"] = None,
        location: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword identity:
        :paramtype identity: ~azure.mgmt.nginx.models.IdentityProperties
        :keyword properties:
        :paramtype properties: ~azure.mgmt.nginx.models.NginxDeploymentProperties
        :keyword tags: A set of tags. Dictionary of :code:`<string>`.
        :paramtype tags: dict[str, str]
        :keyword sku:
        :paramtype sku: ~azure.mgmt.nginx.models.ResourceSku
        :keyword location:
        :paramtype location: str
        """
        super(NginxDeployment, self).__init__(**kwargs)
        self.id = None
        self.name = None
        self.type = None
        self.identity = identity
        self.properties = properties
        self.tags = tags
        self.sku = sku
        self.location = location
        self.system_data = None


class NginxDeploymentListResponse(msrest.serialization.Model):
    """NginxDeploymentListResponse.

    :ivar value:
    :vartype value: list[~azure.mgmt.nginx.models.NginxDeployment]
    :ivar next_link:
    :vartype next_link: str
    """

    _attribute_map = {
        'value': {'key': 'value', 'type': '[NginxDeployment]'},
        'next_link': {'key': 'nextLink', 'type': 'str'},
    }

    def __init__(
        self,
        *,
        value: Optional[List["NginxDeployment"]] = None,
        next_link: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword value:
        :paramtype value: list[~azure.mgmt.nginx.models.NginxDeployment]
        :keyword next_link:
        :paramtype next_link: str
        """
        super(NginxDeploymentListResponse, self).__init__(**kwargs)
        self.value = value
        self.next_link = next_link


class NginxDeploymentProperties(msrest.serialization.Model):
    """NginxDeploymentProperties.

    Variables are only populated by the server, and will be ignored when sending a request.

    :ivar provisioning_state: Possible values include: "Accepted", "Creating", "Updating",
     "Deleting", "Succeeded", "Failed", "Canceled", "Deleted", "NotSpecified".
    :vartype provisioning_state: str or ~azure.mgmt.nginx.models.ProvisioningState
    :ivar nginx_version:
    :vartype nginx_version: str
    :ivar managed_resource_group: The managed resource group to deploy VNet injection related
     network resources.
    :vartype managed_resource_group: str
    :ivar network_profile:
    :vartype network_profile: ~azure.mgmt.nginx.models.NginxNetworkProfile
    :ivar ip_address: The IP address of the deployment.
    :vartype ip_address: str
    :ivar enable_diagnostics_support:
    :vartype enable_diagnostics_support: bool
    """

    _validation = {
        'nginx_version': {'readonly': True},
        'ip_address': {'readonly': True},
    }

    _attribute_map = {
        'provisioning_state': {'key': 'provisioningState', 'type': 'str'},
        'nginx_version': {'key': 'nginxVersion', 'type': 'str'},
        'managed_resource_group': {'key': 'managedResourceGroup', 'type': 'str'},
        'network_profile': {'key': 'networkProfile', 'type': 'NginxNetworkProfile'},
        'ip_address': {'key': 'ipAddress', 'type': 'str'},
        'enable_diagnostics_support': {'key': 'enableDiagnosticsSupport', 'type': 'bool'},
    }

    def __init__(
        self,
        *,
        provisioning_state: Optional[Union[str, "ProvisioningState"]] = None,
        managed_resource_group: Optional[str] = None,
        network_profile: Optional["NginxNetworkProfile"] = None,
        enable_diagnostics_support: Optional[bool] = None,
        **kwargs
    ):
        """
        :keyword provisioning_state: Possible values include: "Accepted", "Creating", "Updating",
         "Deleting", "Succeeded", "Failed", "Canceled", "Deleted", "NotSpecified".
        :paramtype provisioning_state: str or ~azure.mgmt.nginx.models.ProvisioningState
        :keyword managed_resource_group: The managed resource group to deploy VNet injection related
         network resources.
        :paramtype managed_resource_group: str
        :keyword network_profile:
        :paramtype network_profile: ~azure.mgmt.nginx.models.NginxNetworkProfile
        :keyword enable_diagnostics_support:
        :paramtype enable_diagnostics_support: bool
        """
        super(NginxDeploymentProperties, self).__init__(**kwargs)
        self.provisioning_state = provisioning_state
        self.nginx_version = None
        self.managed_resource_group = managed_resource_group
        self.network_profile = network_profile
        self.ip_address = None
        self.enable_diagnostics_support = enable_diagnostics_support


class NginxDeploymentUpdateParameters(msrest.serialization.Model):
    """NginxDeploymentUpdateParameters.

    :ivar identity:
    :vartype identity: ~azure.mgmt.nginx.models.IdentityProperties
    :ivar tags: A set of tags. Dictionary of :code:`<string>`.
    :vartype tags: dict[str, str]
    :ivar sku:
    :vartype sku: ~azure.mgmt.nginx.models.ResourceSku
    :ivar location:
    :vartype location: str
    :ivar properties:
    :vartype properties: ~azure.mgmt.nginx.models.NginxDeploymentUpdateProperties
    """

    _attribute_map = {
        'identity': {'key': 'identity', 'type': 'IdentityProperties'},
        'tags': {'key': 'tags', 'type': '{str}'},
        'sku': {'key': 'sku', 'type': 'ResourceSku'},
        'location': {'key': 'location', 'type': 'str'},
        'properties': {'key': 'properties', 'type': 'NginxDeploymentUpdateProperties'},
    }

    def __init__(
        self,
        *,
        identity: Optional["IdentityProperties"] = None,
        tags: Optional[Dict[str, str]] = None,
        sku: Optional["ResourceSku"] = None,
        location: Optional[str] = None,
        properties: Optional["NginxDeploymentUpdateProperties"] = None,
        **kwargs
    ):
        """
        :keyword identity:
        :paramtype identity: ~azure.mgmt.nginx.models.IdentityProperties
        :keyword tags: A set of tags. Dictionary of :code:`<string>`.
        :paramtype tags: dict[str, str]
        :keyword sku:
        :paramtype sku: ~azure.mgmt.nginx.models.ResourceSku
        :keyword location:
        :paramtype location: str
        :keyword properties:
        :paramtype properties: ~azure.mgmt.nginx.models.NginxDeploymentUpdateProperties
        """
        super(NginxDeploymentUpdateParameters, self).__init__(**kwargs)
        self.identity = identity
        self.tags = tags
        self.sku = sku
        self.location = location
        self.properties = properties


class NginxDeploymentUpdateProperties(msrest.serialization.Model):
    """NginxDeploymentUpdateProperties.

    :ivar enable_diagnostics_support:
    :vartype enable_diagnostics_support: bool
    """

    _attribute_map = {
        'enable_diagnostics_support': {'key': 'enableDiagnosticsSupport', 'type': 'bool'},
    }

    def __init__(
        self,
        *,
        enable_diagnostics_support: Optional[bool] = None,
        **kwargs
    ):
        """
        :keyword enable_diagnostics_support:
        :paramtype enable_diagnostics_support: bool
        """
        super(NginxDeploymentUpdateProperties, self).__init__(**kwargs)
        self.enable_diagnostics_support = enable_diagnostics_support


class NginxFrontendIPConfiguration(msrest.serialization.Model):
    """NginxFrontendIPConfiguration.

    :ivar public_ip_addresses:
    :vartype public_ip_addresses: list[~azure.mgmt.nginx.models.NginxPublicIPAddress]
    :ivar private_ip_addresses:
    :vartype private_ip_addresses: list[~azure.mgmt.nginx.models.NginxPrivateIPAddress]
    """

    _attribute_map = {
        'public_ip_addresses': {'key': 'publicIPAddresses', 'type': '[NginxPublicIPAddress]'},
        'private_ip_addresses': {'key': 'privateIPAddresses', 'type': '[NginxPrivateIPAddress]'},
    }

    def __init__(
        self,
        *,
        public_ip_addresses: Optional[List["NginxPublicIPAddress"]] = None,
        private_ip_addresses: Optional[List["NginxPrivateIPAddress"]] = None,
        **kwargs
    ):
        """
        :keyword public_ip_addresses:
        :paramtype public_ip_addresses: list[~azure.mgmt.nginx.models.NginxPublicIPAddress]
        :keyword private_ip_addresses:
        :paramtype private_ip_addresses: list[~azure.mgmt.nginx.models.NginxPrivateIPAddress]
        """
        super(NginxFrontendIPConfiguration, self).__init__(**kwargs)
        self.public_ip_addresses = public_ip_addresses
        self.private_ip_addresses = private_ip_addresses


class NginxNetworkInterfaceConfiguration(msrest.serialization.Model):
    """NginxNetworkInterfaceConfiguration.

    :ivar subnet_id:
    :vartype subnet_id: str
    """

    _attribute_map = {
        'subnet_id': {'key': 'subnetId', 'type': 'str'},
    }

    def __init__(
        self,
        *,
        subnet_id: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword subnet_id:
        :paramtype subnet_id: str
        """
        super(NginxNetworkInterfaceConfiguration, self).__init__(**kwargs)
        self.subnet_id = subnet_id


class NginxNetworkProfile(msrest.serialization.Model):
    """NginxNetworkProfile.

    :ivar front_end_ip_configuration:
    :vartype front_end_ip_configuration: ~azure.mgmt.nginx.models.NginxFrontendIPConfiguration
    :ivar network_interface_configuration:
    :vartype network_interface_configuration:
     ~azure.mgmt.nginx.models.NginxNetworkInterfaceConfiguration
    """

    _attribute_map = {
        'front_end_ip_configuration': {'key': 'frontEndIPConfiguration', 'type': 'NginxFrontendIPConfiguration'},
        'network_interface_configuration': {'key': 'networkInterfaceConfiguration', 'type': 'NginxNetworkInterfaceConfiguration'},
    }

    def __init__(
        self,
        *,
        front_end_ip_configuration: Optional["NginxFrontendIPConfiguration"] = None,
        network_interface_configuration: Optional["NginxNetworkInterfaceConfiguration"] = None,
        **kwargs
    ):
        """
        :keyword front_end_ip_configuration:
        :paramtype front_end_ip_configuration: ~azure.mgmt.nginx.models.NginxFrontendIPConfiguration
        :keyword network_interface_configuration:
        :paramtype network_interface_configuration:
         ~azure.mgmt.nginx.models.NginxNetworkInterfaceConfiguration
        """
        super(NginxNetworkProfile, self).__init__(**kwargs)
        self.front_end_ip_configuration = front_end_ip_configuration
        self.network_interface_configuration = network_interface_configuration


class NginxPrivateIPAddress(msrest.serialization.Model):
    """NginxPrivateIPAddress.

    :ivar private_ip_address:
    :vartype private_ip_address: str
    :ivar private_ip_allocation_method: Possible values include: "Static", "Dynamic".
    :vartype private_ip_allocation_method: str or
     ~azure.mgmt.nginx.models.NginxPrivateIPAllocationMethod
    :ivar subnet_id:
    :vartype subnet_id: str
    """

    _attribute_map = {
        'private_ip_address': {'key': 'privateIPAddress', 'type': 'str'},
        'private_ip_allocation_method': {'key': 'privateIPAllocationMethod', 'type': 'str'},
        'subnet_id': {'key': 'subnetId', 'type': 'str'},
    }

    def __init__(
        self,
        *,
        private_ip_address: Optional[str] = None,
        private_ip_allocation_method: Optional[Union[str, "NginxPrivateIPAllocationMethod"]] = None,
        subnet_id: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword private_ip_address:
        :paramtype private_ip_address: str
        :keyword private_ip_allocation_method: Possible values include: "Static", "Dynamic".
        :paramtype private_ip_allocation_method: str or
         ~azure.mgmt.nginx.models.NginxPrivateIPAllocationMethod
        :keyword subnet_id:
        :paramtype subnet_id: str
        """
        super(NginxPrivateIPAddress, self).__init__(**kwargs)
        self.private_ip_address = private_ip_address
        self.private_ip_allocation_method = private_ip_allocation_method
        self.subnet_id = subnet_id


class NginxPublicIPAddress(msrest.serialization.Model):
    """NginxPublicIPAddress.

    :ivar id:
    :vartype id: str
    """

    _attribute_map = {
        'id': {'key': 'id', 'type': 'str'},
    }

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword id:
        :paramtype id: str
        """
        super(NginxPublicIPAddress, self).__init__(**kwargs)
        self.id = id


class OperationDisplay(msrest.serialization.Model):
    """The object that represents the operation.

    :ivar provider: Service provider: Nginx.NginxPlus.
    :vartype provider: str
    :ivar resource: Type on which the operation is performed, e.g., 'deployments'.
    :vartype resource: str
    :ivar operation: Operation type, e.g., read, write, delete, etc.
    :vartype operation: str
    :ivar description: Description of the operation, e.g., 'Write deployments'.
    :vartype description: str
    """

    _attribute_map = {
        'provider': {'key': 'provider', 'type': 'str'},
        'resource': {'key': 'resource', 'type': 'str'},
        'operation': {'key': 'operation', 'type': 'str'},
        'description': {'key': 'description', 'type': 'str'},
    }

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        resource: Optional[str] = None,
        operation: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword provider: Service provider: Nginx.NginxPlus.
        :paramtype provider: str
        :keyword resource: Type on which the operation is performed, e.g., 'deployments'.
        :paramtype resource: str
        :keyword operation: Operation type, e.g., read, write, delete, etc.
        :paramtype operation: str
        :keyword description: Description of the operation, e.g., 'Write deployments'.
        :paramtype description: str
        """
        super(OperationDisplay, self).__init__(**kwargs)
        self.provider = provider
        self.resource = resource
        self.operation = operation
        self.description = description


class OperationListResult(msrest.serialization.Model):
    """Result of GET request to list Nginx.NginxPlus operations.

    :ivar value: List of operations supported by the Nginx.NginxPlus provider.
    :vartype value: list[~azure.mgmt.nginx.models.OperationResult]
    :ivar next_link: URL to get the next set of operation list results if there are any.
    :vartype next_link: str
    """

    _attribute_map = {
        'value': {'key': 'value', 'type': '[OperationResult]'},
        'next_link': {'key': 'nextLink', 'type': 'str'},
    }

    def __init__(
        self,
        *,
        value: Optional[List["OperationResult"]] = None,
        next_link: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword value: List of operations supported by the Nginx.NginxPlus provider.
        :paramtype value: list[~azure.mgmt.nginx.models.OperationResult]
        :keyword next_link: URL to get the next set of operation list results if there are any.
        :paramtype next_link: str
        """
        super(OperationListResult, self).__init__(**kwargs)
        self.value = value
        self.next_link = next_link


class OperationResult(msrest.serialization.Model):
    """A Nginx.NginxPlus REST API operation.

    :ivar name: Operation name: {provider}/{resource}/{operation}.
    :vartype name: str
    :ivar display: The object that represents the operation.
    :vartype display: ~azure.mgmt.nginx.models.OperationDisplay
    :ivar is_data_action: Indicates whether the operation is a data action.
    :vartype is_data_action: bool
    """

    _attribute_map = {
        'name': {'key': 'name', 'type': 'str'},
        'display': {'key': 'display', 'type': 'OperationDisplay'},
        'is_data_action': {'key': 'isDataAction', 'type': 'bool'},
    }

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        display: Optional["OperationDisplay"] = None,
        is_data_action: Optional[bool] = None,
        **kwargs
    ):
        """
        :keyword name: Operation name: {provider}/{resource}/{operation}.
        :paramtype name: str
        :keyword display: The object that represents the operation.
        :paramtype display: ~azure.mgmt.nginx.models.OperationDisplay
        :keyword is_data_action: Indicates whether the operation is a data action.
        :paramtype is_data_action: bool
        """
        super(OperationResult, self).__init__(**kwargs)
        self.name = name
        self.display = display
        self.is_data_action = is_data_action


class ResourceProviderDefaultErrorResponse(msrest.serialization.Model):
    """ResourceProviderDefaultErrorResponse.

    :ivar error:
    :vartype error: ~azure.mgmt.nginx.models.ErrorResponseBody
    """

    _attribute_map = {
        'error': {'key': 'error', 'type': 'ErrorResponseBody'},
    }

    def __init__(
        self,
        *,
        error: Optional["ErrorResponseBody"] = None,
        **kwargs
    ):
        """
        :keyword error:
        :paramtype error: ~azure.mgmt.nginx.models.ErrorResponseBody
        """
        super(ResourceProviderDefaultErrorResponse, self).__init__(**kwargs)
        self.error = error


class ResourceSku(msrest.serialization.Model):
    """ResourceSku.

    All required parameters must be populated in order to send to Azure.

    :ivar name: Required. Name of the SKU.
    :vartype name: str
    """

    _validation = {
        'name': {'required': True},
    }

    _attribute_map = {
        'name': {'key': 'name', 'type': 'str'},
    }

    def __init__(
        self,
        *,
        name: str,
        **kwargs
    ):
        """
        :keyword name: Required. Name of the SKU.
        :paramtype name: str
        """
        super(ResourceSku, self).__init__(**kwargs)
        self.name = name


class SystemData(msrest.serialization.Model):
    """Metadata pertaining to creation and last modification of the resource.

    :ivar created_by: The identity that created the resource.
    :vartype created_by: str
    :ivar created_by_type: The type of identity that created the resource. Possible values include:
     "User", "Application", "ManagedIdentity", "Key".
    :vartype created_by_type: str or ~azure.mgmt.nginx.models.CreatedByType
    :ivar created_at: The timestamp of resource creation (UTC).
    :vartype created_at: ~datetime.datetime
    :ivar last_modified_by: The identity that last modified the resource.
    :vartype last_modified_by: str
    :ivar last_modified_by_type: The type of identity that last modified the resource. Possible
     values include: "User", "Application", "ManagedIdentity", "Key".
    :vartype last_modified_by_type: str or ~azure.mgmt.nginx.models.CreatedByType
    :ivar last_modified_at: The timestamp of resource last modification (UTC).
    :vartype last_modified_at: ~datetime.datetime
    """

    _attribute_map = {
        'created_by': {'key': 'createdBy', 'type': 'str'},
        'created_by_type': {'key': 'createdByType', 'type': 'str'},
        'created_at': {'key': 'createdAt', 'type': 'iso-8601'},
        'last_modified_by': {'key': 'lastModifiedBy', 'type': 'str'},
        'last_modified_by_type': {'key': 'lastModifiedByType', 'type': 'str'},
        'last_modified_at': {'key': 'lastModifiedAt', 'type': 'iso-8601'},
    }

    def __init__(
        self,
        *,
        created_by: Optional[str] = None,
        created_by_type: Optional[Union[str, "CreatedByType"]] = None,
        created_at: Optional[datetime.datetime] = None,
        last_modified_by: Optional[str] = None,
        last_modified_by_type: Optional[Union[str, "CreatedByType"]] = None,
        last_modified_at: Optional[datetime.datetime] = None,
        **kwargs
    ):
        """
        :keyword created_by: The identity that created the resource.
        :paramtype created_by: str
        :keyword created_by_type: The type of identity that created the resource. Possible values
         include: "User", "Application", "ManagedIdentity", "Key".
        :paramtype created_by_type: str or ~azure.mgmt.nginx.models.CreatedByType
        :keyword created_at: The timestamp of resource creation (UTC).
        :paramtype created_at: ~datetime.datetime
        :keyword last_modified_by: The identity that last modified the resource.
        :paramtype last_modified_by: str
        :keyword last_modified_by_type: The type of identity that last modified the resource. Possible
         values include: "User", "Application", "ManagedIdentity", "Key".
        :paramtype last_modified_by_type: str or ~azure.mgmt.nginx.models.CreatedByType
        :keyword last_modified_at: The timestamp of resource last modification (UTC).
        :paramtype last_modified_at: ~datetime.datetime
        """
        super(SystemData, self).__init__(**kwargs)
        self.created_by = created_by
        self.created_by_type = created_by_type
        self.created_at = created_at
        self.last_modified_by = last_modified_by
        self.last_modified_by_type = last_modified_by_type
        self.last_modified_at = last_modified_at


class UserIdentityProperties(msrest.serialization.Model):
    """UserIdentityProperties.

    Variables are only populated by the server, and will be ignored when sending a request.

    :ivar principal_id:
    :vartype principal_id: str
    :ivar client_id:
    :vartype client_id: str
    """

    _validation = {
        'principal_id': {'readonly': True},
        'client_id': {'readonly': True},
    }

    _attribute_map = {
        'principal_id': {'key': 'principalId', 'type': 'str'},
        'client_id': {'key': 'clientId', 'type': 'str'},
    }

    def __init__(
        self,
        **kwargs
    ):
        """
        """
        super(UserIdentityProperties, self).__init__(**kwargs)
        self.principal_id = None
        self.client_id = None
