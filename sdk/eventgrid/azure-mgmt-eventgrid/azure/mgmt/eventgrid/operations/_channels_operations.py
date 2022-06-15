# pylint: disable=too-many-lines
# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
from typing import Any, Callable, Dict, Iterable, Optional, TypeVar, Union

from msrest import Serializer

from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, map_error
from azure.core.paging import ItemPaged
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.polling import LROPoller, NoPolling, PollingMethod
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.mgmt.core.exceptions import ARMErrorFormat
from azure.mgmt.core.polling.arm_polling import ARMPolling

from .. import models as _models
from .._vendor import _convert_request, _format_url_section
T = TypeVar('T')
JSONType = Any
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]

_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_get_request(
    subscription_id: str,
    resource_group_name: str,
    partner_namespace_name: str,
    channel_name: str,
    **kwargs: Any
) -> HttpRequest:
    api_version = kwargs.pop('api_version', "2021-10-15-preview")  # type: str

    accept = "application/json"
    # Construct URL
    _url = kwargs.pop("template_url", "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.EventGrid/partnerNamespaces/{partnerNamespaceName}/channels/{channelName}")  # pylint: disable=line-too-long
    path_format_arguments = {
        "subscriptionId": _SERIALIZER.url("subscription_id", subscription_id, 'str'),
        "resourceGroupName": _SERIALIZER.url("resource_group_name", resource_group_name, 'str'),
        "partnerNamespaceName": _SERIALIZER.url("partner_namespace_name", partner_namespace_name, 'str'),
        "channelName": _SERIALIZER.url("channel_name", channel_name, 'str'),
    }

    _url = _format_url_section(_url, **path_format_arguments)

    # Construct parameters
    _query_parameters = kwargs.pop("params", {})  # type: Dict[str, Any]
    _query_parameters['api-version'] = _SERIALIZER.query("api_version", api_version, 'str')

    # Construct headers
    _header_parameters = kwargs.pop("headers", {})  # type: Dict[str, Any]
    _header_parameters['Accept'] = _SERIALIZER.header("accept", accept, 'str')

    return HttpRequest(
        method="GET",
        url=_url,
        params=_query_parameters,
        headers=_header_parameters,
        **kwargs
    )


def build_create_or_update_request(
    subscription_id: str,
    resource_group_name: str,
    partner_namespace_name: str,
    channel_name: str,
    *,
    json: JSONType = None,
    content: Any = None,
    **kwargs: Any
) -> HttpRequest:
    api_version = kwargs.pop('api_version', "2021-10-15-preview")  # type: str
    content_type = kwargs.pop('content_type', None)  # type: Optional[str]

    accept = "application/json"
    # Construct URL
    _url = kwargs.pop("template_url", "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.EventGrid/partnerNamespaces/{partnerNamespaceName}/channels/{channelName}")  # pylint: disable=line-too-long
    path_format_arguments = {
        "subscriptionId": _SERIALIZER.url("subscription_id", subscription_id, 'str'),
        "resourceGroupName": _SERIALIZER.url("resource_group_name", resource_group_name, 'str'),
        "partnerNamespaceName": _SERIALIZER.url("partner_namespace_name", partner_namespace_name, 'str'),
        "channelName": _SERIALIZER.url("channel_name", channel_name, 'str'),
    }

    _url = _format_url_section(_url, **path_format_arguments)

    # Construct parameters
    _query_parameters = kwargs.pop("params", {})  # type: Dict[str, Any]
    _query_parameters['api-version'] = _SERIALIZER.query("api_version", api_version, 'str')

    # Construct headers
    _header_parameters = kwargs.pop("headers", {})  # type: Dict[str, Any]
    if content_type is not None:
        _header_parameters['Content-Type'] = _SERIALIZER.header("content_type", content_type, 'str')
    _header_parameters['Accept'] = _SERIALIZER.header("accept", accept, 'str')

    return HttpRequest(
        method="PUT",
        url=_url,
        params=_query_parameters,
        headers=_header_parameters,
        json=json,
        content=content,
        **kwargs
    )


def build_delete_request_initial(
    subscription_id: str,
    resource_group_name: str,
    partner_namespace_name: str,
    channel_name: str,
    **kwargs: Any
) -> HttpRequest:
    api_version = kwargs.pop('api_version', "2021-10-15-preview")  # type: str

    # Construct URL
    _url = kwargs.pop("template_url", "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.EventGrid/partnerNamespaces/{partnerNamespaceName}/channels/{channelName}")  # pylint: disable=line-too-long
    path_format_arguments = {
        "subscriptionId": _SERIALIZER.url("subscription_id", subscription_id, 'str'),
        "resourceGroupName": _SERIALIZER.url("resource_group_name", resource_group_name, 'str'),
        "partnerNamespaceName": _SERIALIZER.url("partner_namespace_name", partner_namespace_name, 'str'),
        "channelName": _SERIALIZER.url("channel_name", channel_name, 'str'),
    }

    _url = _format_url_section(_url, **path_format_arguments)

    # Construct parameters
    _query_parameters = kwargs.pop("params", {})  # type: Dict[str, Any]
    _query_parameters['api-version'] = _SERIALIZER.query("api_version", api_version, 'str')

    return HttpRequest(
        method="DELETE",
        url=_url,
        params=_query_parameters,
        **kwargs
    )


def build_update_request(
    subscription_id: str,
    resource_group_name: str,
    partner_namespace_name: str,
    channel_name: str,
    *,
    json: JSONType = None,
    content: Any = None,
    **kwargs: Any
) -> HttpRequest:
    api_version = kwargs.pop('api_version', "2021-10-15-preview")  # type: str
    content_type = kwargs.pop('content_type', None)  # type: Optional[str]

    # Construct URL
    _url = kwargs.pop("template_url", "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.EventGrid/partnerNamespaces/{partnerNamespaceName}/channels/{channelName}")  # pylint: disable=line-too-long
    path_format_arguments = {
        "subscriptionId": _SERIALIZER.url("subscription_id", subscription_id, 'str'),
        "resourceGroupName": _SERIALIZER.url("resource_group_name", resource_group_name, 'str'),
        "partnerNamespaceName": _SERIALIZER.url("partner_namespace_name", partner_namespace_name, 'str'),
        "channelName": _SERIALIZER.url("channel_name", channel_name, 'str'),
    }

    _url = _format_url_section(_url, **path_format_arguments)

    # Construct parameters
    _query_parameters = kwargs.pop("params", {})  # type: Dict[str, Any]
    _query_parameters['api-version'] = _SERIALIZER.query("api_version", api_version, 'str')

    # Construct headers
    _header_parameters = kwargs.pop("headers", {})  # type: Dict[str, Any]
    if content_type is not None:
        _header_parameters['Content-Type'] = _SERIALIZER.header("content_type", content_type, 'str')

    return HttpRequest(
        method="PATCH",
        url=_url,
        params=_query_parameters,
        headers=_header_parameters,
        json=json,
        content=content,
        **kwargs
    )


def build_list_by_partner_namespace_request(
    subscription_id: str,
    resource_group_name: str,
    partner_namespace_name: str,
    *,
    filter: Optional[str] = None,
    top: Optional[int] = None,
    **kwargs: Any
) -> HttpRequest:
    api_version = kwargs.pop('api_version', "2021-10-15-preview")  # type: str

    accept = "application/json"
    # Construct URL
    _url = kwargs.pop("template_url", "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.EventGrid/partnerNamespaces/{partnerNamespaceName}/channels")  # pylint: disable=line-too-long
    path_format_arguments = {
        "subscriptionId": _SERIALIZER.url("subscription_id", subscription_id, 'str'),
        "resourceGroupName": _SERIALIZER.url("resource_group_name", resource_group_name, 'str'),
        "partnerNamespaceName": _SERIALIZER.url("partner_namespace_name", partner_namespace_name, 'str'),
    }

    _url = _format_url_section(_url, **path_format_arguments)

    # Construct parameters
    _query_parameters = kwargs.pop("params", {})  # type: Dict[str, Any]
    _query_parameters['api-version'] = _SERIALIZER.query("api_version", api_version, 'str')
    if filter is not None:
        _query_parameters['$filter'] = _SERIALIZER.query("filter", filter, 'str')
    if top is not None:
        _query_parameters['$top'] = _SERIALIZER.query("top", top, 'int')

    # Construct headers
    _header_parameters = kwargs.pop("headers", {})  # type: Dict[str, Any]
    _header_parameters['Accept'] = _SERIALIZER.header("accept", accept, 'str')

    return HttpRequest(
        method="GET",
        url=_url,
        params=_query_parameters,
        headers=_header_parameters,
        **kwargs
    )


def build_get_full_url_request(
    subscription_id: str,
    resource_group_name: str,
    partner_namespace_name: str,
    channel_name: str,
    **kwargs: Any
) -> HttpRequest:
    api_version = kwargs.pop('api_version', "2021-10-15-preview")  # type: str

    accept = "application/json"
    # Construct URL
    _url = kwargs.pop("template_url", "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.EventGrid/partnerNamespaces/{partnerNamespaceName}/channels/{channelName}/getFullUrl")  # pylint: disable=line-too-long
    path_format_arguments = {
        "subscriptionId": _SERIALIZER.url("subscription_id", subscription_id, 'str'),
        "resourceGroupName": _SERIALIZER.url("resource_group_name", resource_group_name, 'str'),
        "partnerNamespaceName": _SERIALIZER.url("partner_namespace_name", partner_namespace_name, 'str'),
        "channelName": _SERIALIZER.url("channel_name", channel_name, 'str'),
    }

    _url = _format_url_section(_url, **path_format_arguments)

    # Construct parameters
    _query_parameters = kwargs.pop("params", {})  # type: Dict[str, Any]
    _query_parameters['api-version'] = _SERIALIZER.query("api_version", api_version, 'str')

    # Construct headers
    _header_parameters = kwargs.pop("headers", {})  # type: Dict[str, Any]
    _header_parameters['Accept'] = _SERIALIZER.header("accept", accept, 'str')

    return HttpRequest(
        method="POST",
        url=_url,
        params=_query_parameters,
        headers=_header_parameters,
        **kwargs
    )

class ChannelsOperations(object):
    """ChannelsOperations operations.

    You should not instantiate this class directly. Instead, you should create a Client instance that
    instantiates it for you and attaches it as an attribute.

    :ivar models: Alias to model classes used in this operation group.
    :type models: ~azure.mgmt.eventgrid.models
    :param client: Client for service requests.
    :param config: Configuration of service client.
    :param serializer: An object model serializer.
    :param deserializer: An object model deserializer.
    """

    models = _models

    def __init__(self, client, config, serializer, deserializer):
        self._client = client
        self._serialize = serializer
        self._deserialize = deserializer
        self._config = config

    @distributed_trace
    def get(
        self,
        resource_group_name: str,
        partner_namespace_name: str,
        channel_name: str,
        **kwargs: Any
    ) -> "_models.Channel":
        """Get a channel.

        Get properties of a channel.

        :param resource_group_name: The name of the resource group within the partners subscription.
        :type resource_group_name: str
        :param partner_namespace_name: Name of the partner namespace.
        :type partner_namespace_name: str
        :param channel_name: Name of the channel.
        :type channel_name: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: Channel, or the result of cls(response)
        :rtype: ~azure.mgmt.eventgrid.models.Channel
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        cls = kwargs.pop('cls', None)  # type: ClsType["_models.Channel"]
        error_map = {
            401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError
        }
        error_map.update(kwargs.pop('error_map', {}))

        api_version = kwargs.pop('api_version', "2021-10-15-preview")  # type: str

        
        request = build_get_request(
            subscription_id=self._config.subscription_id,
            resource_group_name=resource_group_name,
            partner_namespace_name=partner_namespace_name,
            channel_name=channel_name,
            api_version=api_version,
            template_url=self.get.metadata['url'],
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        pipeline_response = self._client._pipeline.run(  # pylint: disable=protected-access
            request,
            stream=False,
            **kwargs
        )
        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        deserialized = self._deserialize('Channel', pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})

        return deserialized

    get.metadata = {'url': "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.EventGrid/partnerNamespaces/{partnerNamespaceName}/channels/{channelName}"}  # type: ignore


    @distributed_trace
    def create_or_update(
        self,
        resource_group_name: str,
        partner_namespace_name: str,
        channel_name: str,
        channel_info: "_models.Channel",
        **kwargs: Any
    ) -> "_models.Channel":
        """Create or update a channel.

        Synchronously creates or updates a new channel with the specified parameters.

        :param resource_group_name: The name of the resource group within the partners subscription.
        :type resource_group_name: str
        :param partner_namespace_name: Name of the partner namespace.
        :type partner_namespace_name: str
        :param channel_name: Name of the channel.
        :type channel_name: str
        :param channel_info: Channel information.
        :type channel_info: ~azure.mgmt.eventgrid.models.Channel
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: Channel, or the result of cls(response)
        :rtype: ~azure.mgmt.eventgrid.models.Channel
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        cls = kwargs.pop('cls', None)  # type: ClsType["_models.Channel"]
        error_map = {
            401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError
        }
        error_map.update(kwargs.pop('error_map', {}))

        api_version = kwargs.pop('api_version', "2021-10-15-preview")  # type: str
        content_type = kwargs.pop('content_type', "application/json")  # type: Optional[str]

        _json = self._serialize.body(channel_info, 'Channel')

        request = build_create_or_update_request(
            subscription_id=self._config.subscription_id,
            resource_group_name=resource_group_name,
            partner_namespace_name=partner_namespace_name,
            channel_name=channel_name,
            api_version=api_version,
            content_type=content_type,
            json=_json,
            template_url=self.create_or_update.metadata['url'],
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        pipeline_response = self._client._pipeline.run(  # pylint: disable=protected-access
            request,
            stream=False,
            **kwargs
        )
        response = pipeline_response.http_response

        if response.status_code not in [200, 201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        if response.status_code == 200:
            deserialized = self._deserialize('Channel', pipeline_response)

        if response.status_code == 201:
            deserialized = self._deserialize('Channel', pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})

        return deserialized

    create_or_update.metadata = {'url': "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.EventGrid/partnerNamespaces/{partnerNamespaceName}/channels/{channelName}"}  # type: ignore


    def _delete_initial(  # pylint: disable=inconsistent-return-statements
        self,
        resource_group_name: str,
        partner_namespace_name: str,
        channel_name: str,
        **kwargs: Any
    ) -> None:
        cls = kwargs.pop('cls', None)  # type: ClsType[None]
        error_map = {
            401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError
        }
        error_map.update(kwargs.pop('error_map', {}))

        api_version = kwargs.pop('api_version', "2021-10-15-preview")  # type: str

        
        request = build_delete_request_initial(
            subscription_id=self._config.subscription_id,
            resource_group_name=resource_group_name,
            partner_namespace_name=partner_namespace_name,
            channel_name=channel_name,
            api_version=api_version,
            template_url=self._delete_initial.metadata['url'],
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        pipeline_response = self._client._pipeline.run(  # pylint: disable=protected-access
            request,
            stream=False,
            **kwargs
        )
        response = pipeline_response.http_response

        if response.status_code not in [200, 202, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        if cls:
            return cls(pipeline_response, None, {})

    _delete_initial.metadata = {'url': "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.EventGrid/partnerNamespaces/{partnerNamespaceName}/channels/{channelName}"}  # type: ignore


    @distributed_trace
    def begin_delete(  # pylint: disable=inconsistent-return-statements
        self,
        resource_group_name: str,
        partner_namespace_name: str,
        channel_name: str,
        **kwargs: Any
    ) -> LROPoller[None]:
        """Delete a channel.

        Delete an existing channel.

        :param resource_group_name: The name of the resource group within the partners subscription.
        :type resource_group_name: str
        :param partner_namespace_name: Name of the partner namespace.
        :type partner_namespace_name: str
        :param channel_name: Name of the channel.
        :type channel_name: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :keyword str continuation_token: A continuation token to restart a poller from a saved state.
        :keyword polling: By default, your polling method will be ARMPolling. Pass in False for this
         operation to not poll, or pass in your own initialized polling object for a personal polling
         strategy.
        :paramtype polling: bool or ~azure.core.polling.PollingMethod
        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no
         Retry-After header is present.
        :return: An instance of LROPoller that returns either None or the result of cls(response)
        :rtype: ~azure.core.polling.LROPoller[None]
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        api_version = kwargs.pop('api_version', "2021-10-15-preview")  # type: str
        polling = kwargs.pop('polling', True)  # type: Union[bool, PollingMethod]
        cls = kwargs.pop('cls', None)  # type: ClsType[None]
        lro_delay = kwargs.pop(
            'polling_interval',
            self._config.polling_interval
        )
        cont_token = kwargs.pop('continuation_token', None)  # type: Optional[str]
        if cont_token is None:
            raw_result = self._delete_initial(
                resource_group_name=resource_group_name,
                partner_namespace_name=partner_namespace_name,
                channel_name=channel_name,
                api_version=api_version,
                cls=lambda x,y,z: x,
                **kwargs
            )
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if cls:
                return cls(pipeline_response, None, {})


        if polling is True: polling_method = ARMPolling(lro_delay, **kwargs)
        elif polling is False: polling_method = NoPolling()
        else: polling_method = polling
        if cont_token:
            return LROPoller.from_continuation_token(
                polling_method=polling_method,
                continuation_token=cont_token,
                client=self._client,
                deserialization_callback=get_long_running_output
            )
        return LROPoller(self._client, raw_result, get_long_running_output, polling_method)

    begin_delete.metadata = {'url': "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.EventGrid/partnerNamespaces/{partnerNamespaceName}/channels/{channelName}"}  # type: ignore

    @distributed_trace
    def update(  # pylint: disable=inconsistent-return-statements
        self,
        resource_group_name: str,
        partner_namespace_name: str,
        channel_name: str,
        channel_update_parameters: "_models.ChannelUpdateParameters",
        **kwargs: Any
    ) -> None:
        """Update a Channel.

        Synchronously updates a channel with the specified parameters.

        :param resource_group_name: The name of the resource group within the partners subscription.
        :type resource_group_name: str
        :param partner_namespace_name: Name of the partner namespace.
        :type partner_namespace_name: str
        :param channel_name: Name of the channel.
        :type channel_name: str
        :param channel_update_parameters: Channel update information.
        :type channel_update_parameters: ~azure.mgmt.eventgrid.models.ChannelUpdateParameters
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: None, or the result of cls(response)
        :rtype: None
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        cls = kwargs.pop('cls', None)  # type: ClsType[None]
        error_map = {
            401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError
        }
        error_map.update(kwargs.pop('error_map', {}))

        api_version = kwargs.pop('api_version', "2021-10-15-preview")  # type: str
        content_type = kwargs.pop('content_type', "application/json")  # type: Optional[str]

        _json = self._serialize.body(channel_update_parameters, 'ChannelUpdateParameters')

        request = build_update_request(
            subscription_id=self._config.subscription_id,
            resource_group_name=resource_group_name,
            partner_namespace_name=partner_namespace_name,
            channel_name=channel_name,
            api_version=api_version,
            content_type=content_type,
            json=_json,
            template_url=self.update.metadata['url'],
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        pipeline_response = self._client._pipeline.run(  # pylint: disable=protected-access
            request,
            stream=False,
            **kwargs
        )
        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        if cls:
            return cls(pipeline_response, None, {})

    update.metadata = {'url': "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.EventGrid/partnerNamespaces/{partnerNamespaceName}/channels/{channelName}"}  # type: ignore


    @distributed_trace
    def list_by_partner_namespace(
        self,
        resource_group_name: str,
        partner_namespace_name: str,
        filter: Optional[str] = None,
        top: Optional[int] = None,
        **kwargs: Any
    ) -> Iterable["_models.ChannelsListResult"]:
        """List channels.

        List all the channels in a partner namespace.

        :param resource_group_name: The name of the resource group within the partners subscription.
        :type resource_group_name: str
        :param partner_namespace_name: Name of the partner namespace.
        :type partner_namespace_name: str
        :param filter: The query used to filter the search results using OData syntax. Filtering is
         permitted on the 'name' property only and with limited number of OData operations. These
         operations are: the 'contains' function as well as the following logical operations: not, and,
         or, eq (for equal), and ne (for not equal). No arithmetic operations are supported. The
         following is a valid filter example: $filter=contains(namE, 'PATTERN') and name ne 'PATTERN-1'.
         The following is not a valid filter example: $filter=location eq 'westus'. Default value is
         None.
        :type filter: str
        :param top: The number of results to return per page for the list operation. Valid range for
         top parameter is 1 to 100. If not specified, the default number of results to be returned is 20
         items per page. Default value is None.
        :type top: int
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: An iterator like instance of either ChannelsListResult or the result of cls(response)
        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.eventgrid.models.ChannelsListResult]
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        api_version = kwargs.pop('api_version', "2021-10-15-preview")  # type: str

        cls = kwargs.pop('cls', None)  # type: ClsType["_models.ChannelsListResult"]
        error_map = {
            401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError
        }
        error_map.update(kwargs.pop('error_map', {}))
        def prepare_request(next_link=None):
            if not next_link:
                
                request = build_list_by_partner_namespace_request(
                    subscription_id=self._config.subscription_id,
                    resource_group_name=resource_group_name,
                    partner_namespace_name=partner_namespace_name,
                    api_version=api_version,
                    filter=filter,
                    top=top,
                    template_url=self.list_by_partner_namespace.metadata['url'],
                )
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)

            else:
                
                request = build_list_by_partner_namespace_request(
                    subscription_id=self._config.subscription_id,
                    resource_group_name=resource_group_name,
                    partner_namespace_name=partner_namespace_name,
                    api_version=api_version,
                    filter=filter,
                    top=top,
                    template_url=next_link,
                )
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = "GET"
            return request

        def extract_data(pipeline_response):
            deserialized = self._deserialize("ChannelsListResult", pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)
            return deserialized.next_link or None, iter(list_of_elem)

        def get_next(next_link=None):
            request = prepare_request(next_link)

            pipeline_response = self._client._pipeline.run(  # pylint: disable=protected-access
                request,
                stream=False,
                **kwargs
            )
            response = pipeline_response.http_response

            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)

            return pipeline_response


        return ItemPaged(
            get_next, extract_data
        )
    list_by_partner_namespace.metadata = {'url': "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.EventGrid/partnerNamespaces/{partnerNamespaceName}/channels"}  # type: ignore

    @distributed_trace
    def get_full_url(
        self,
        resource_group_name: str,
        partner_namespace_name: str,
        channel_name: str,
        **kwargs: Any
    ) -> "_models.EventSubscriptionFullUrl":
        """Get full URL of partner destination channel.

        Get the full endpoint URL of a partner destination channel.

        :param resource_group_name: The name of the resource group within the partners subscription.
        :type resource_group_name: str
        :param partner_namespace_name: Name of the partner namespace.
        :type partner_namespace_name: str
        :param channel_name: Name of the Channel.
        :type channel_name: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: EventSubscriptionFullUrl, or the result of cls(response)
        :rtype: ~azure.mgmt.eventgrid.models.EventSubscriptionFullUrl
        :raises: ~azure.core.exceptions.HttpResponseError
        """
        cls = kwargs.pop('cls', None)  # type: ClsType["_models.EventSubscriptionFullUrl"]
        error_map = {
            401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError
        }
        error_map.update(kwargs.pop('error_map', {}))

        api_version = kwargs.pop('api_version', "2021-10-15-preview")  # type: str

        
        request = build_get_full_url_request(
            subscription_id=self._config.subscription_id,
            resource_group_name=resource_group_name,
            partner_namespace_name=partner_namespace_name,
            channel_name=channel_name,
            api_version=api_version,
            template_url=self.get_full_url.metadata['url'],
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        pipeline_response = self._client._pipeline.run(  # pylint: disable=protected-access
            request,
            stream=False,
            **kwargs
        )
        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        deserialized = self._deserialize('EventSubscriptionFullUrl', pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})

        return deserialized

    get_full_url.metadata = {'url': "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.EventGrid/partnerNamespaces/{partnerNamespaceName}/channels/{channelName}/getFullUrl"}  # type: ignore

