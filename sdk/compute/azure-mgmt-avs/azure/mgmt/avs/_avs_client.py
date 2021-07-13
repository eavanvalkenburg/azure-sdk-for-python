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

from msrest.service_client import SDKClient
from msrest import Serializer, Deserializer

from ._configuration import AVSClientConfiguration
from .operations import Operations
from .operations import LocationsOperations
from .operations import PrivateCloudsOperations
from .operations import ClustersOperations
from .operations import DatastoresOperations
from .operations import HcxEnterpriseSitesOperations
from .operations import AuthorizationsOperations
from .operations import GlobalReachConnectionsOperations
from .operations import WorkloadNetworksOperations
from .operations import CloudLinksOperations
from .operations import AddonsOperations
from .operations import ScriptPackagesOperations
from .operations import ScriptCmdletsOperations
from .operations import ScriptExecutionsOperations
from . import models


class AVSClient(SDKClient):
    """Azure VMware Solution API

    :ivar config: Configuration for client.
    :vartype config: AVSClientConfiguration

    :ivar operations: Operations operations
    :vartype operations: azure.mgmt.avs.operations.Operations
    :ivar locations: Locations operations
    :vartype locations: azure.mgmt.avs.operations.LocationsOperations
    :ivar private_clouds: PrivateClouds operations
    :vartype private_clouds: azure.mgmt.avs.operations.PrivateCloudsOperations
    :ivar clusters: Clusters operations
    :vartype clusters: azure.mgmt.avs.operations.ClustersOperations
    :ivar datastores: Datastores operations
    :vartype datastores: azure.mgmt.avs.operations.DatastoresOperations
    :ivar hcx_enterprise_sites: HcxEnterpriseSites operations
    :vartype hcx_enterprise_sites: azure.mgmt.avs.operations.HcxEnterpriseSitesOperations
    :ivar authorizations: Authorizations operations
    :vartype authorizations: azure.mgmt.avs.operations.AuthorizationsOperations
    :ivar global_reach_connections: GlobalReachConnections operations
    :vartype global_reach_connections: azure.mgmt.avs.operations.GlobalReachConnectionsOperations
    :ivar workload_networks: WorkloadNetworks operations
    :vartype workload_networks: azure.mgmt.avs.operations.WorkloadNetworksOperations
    :ivar cloud_links: CloudLinks operations
    :vartype cloud_links: azure.mgmt.avs.operations.CloudLinksOperations
    :ivar addons: Addons operations
    :vartype addons: azure.mgmt.avs.operations.AddonsOperations
    :ivar script_packages: ScriptPackages operations
    :vartype script_packages: azure.mgmt.avs.operations.ScriptPackagesOperations
    :ivar script_cmdlets: ScriptCmdlets operations
    :vartype script_cmdlets: azure.mgmt.avs.operations.ScriptCmdletsOperations
    :ivar script_executions: ScriptExecutions operations
    :vartype script_executions: azure.mgmt.avs.operations.ScriptExecutionsOperations

    :param credentials: Credentials needed for the client to connect to Azure.
    :type credentials: :mod:`A msrestazure Credentials
     object<msrestazure.azure_active_directory>`
    :param subscription_id: The ID of the target subscription.
    :type subscription_id: str
    :param str base_url: Service URL
    """

    def __init__(
            self, credentials, subscription_id, base_url=None):

        self.config = AVSClientConfiguration(credentials, subscription_id, base_url)
        super(AVSClient, self).__init__(self.config.credentials, self.config)

        client_models = {k: v for k, v in models.__dict__.items() if isinstance(v, type)}
        self.api_version = '2021-06-01'
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)

        self.operations = Operations(
            self._client, self.config, self._serialize, self._deserialize)
        self.locations = LocationsOperations(
            self._client, self.config, self._serialize, self._deserialize)
        self.private_clouds = PrivateCloudsOperations(
            self._client, self.config, self._serialize, self._deserialize)
        self.clusters = ClustersOperations(
            self._client, self.config, self._serialize, self._deserialize)
        self.datastores = DatastoresOperations(
            self._client, self.config, self._serialize, self._deserialize)
        self.hcx_enterprise_sites = HcxEnterpriseSitesOperations(
            self._client, self.config, self._serialize, self._deserialize)
        self.authorizations = AuthorizationsOperations(
            self._client, self.config, self._serialize, self._deserialize)
        self.global_reach_connections = GlobalReachConnectionsOperations(
            self._client, self.config, self._serialize, self._deserialize)
        self.workload_networks = WorkloadNetworksOperations(
            self._client, self.config, self._serialize, self._deserialize)
        self.cloud_links = CloudLinksOperations(
            self._client, self.config, self._serialize, self._deserialize)
        self.addons = AddonsOperations(
            self._client, self.config, self._serialize, self._deserialize)
        self.script_packages = ScriptPackagesOperations(
            self._client, self.config, self._serialize, self._deserialize)
        self.script_cmdlets = ScriptCmdletsOperations(
            self._client, self.config, self._serialize, self._deserialize)
        self.script_executions = ScriptExecutionsOperations(
            self._client, self.config, self._serialize, self._deserialize)