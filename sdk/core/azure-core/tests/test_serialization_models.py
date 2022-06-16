# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
import json
import datetime
from typing import Any, List, Literal, Dict, Mapping, Sequence, Set, Tuple, Optional, Union, overload
import pytest
import base64
import isodate
from azure.core.serialization import AzureJSONEncoder, Model, rest_field, rest_discriminator

class BasicResource(Model):
    platform_update_domain_count: int = rest_field(name="platformUpdateDomainCount")  # How many times the platform update domain has been counted
    platform_fault_domain_count: int = rest_field(name="platformFaultDomainCount")  # How many times the platform fault domain has been counted
    virtual_machines: List[Any] = rest_field(name="virtualMachines")  # List of virtual machines

    @overload
    def __init__(
        self,
        *,
        platform_update_domain_count: int,
        platform_fault_domain_count: int,
        virtual_machines: List[Any],
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Pet(Model):

    name: str = rest_field()  # my name
    species: str = rest_field()  # my species

    @overload
    def __init__(self, *, name: str, species: str):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def test_model_and_dict_equal():

    dict_response = {
        "platformUpdateDomainCount": 5,
        "platformFaultDomainCount": 3,
        "virtualMachines": []
    }
    model = BasicResource(platform_update_domain_count=5, platform_fault_domain_count=3, virtual_machines=[])

    assert model == dict_response
    assert (
        model.platform_update_domain_count ==
        model["platformUpdateDomainCount"] ==
        dict_response["platformUpdateDomainCount"] ==
        5
    )
    assert (
        model.platform_fault_domain_count ==
        model['platformFaultDomainCount'] ==
        dict_response['platformFaultDomainCount'] ==
        3
    )
    assert (
        model.virtual_machines ==
        model['virtualMachines'] ==
        dict_response['virtualMachines'])

def test_json_roundtrip():
    dict_response = {
        "platformUpdateDomainCount": 5,
        "platformFaultDomainCount": 3,
        "virtualMachines": []
    }
    model = BasicResource(platform_update_domain_count=5, platform_fault_domain_count=3, virtual_machines=[])
    with pytest.raises(TypeError):
        json.dumps(model)
    assert json.dumps(dict(model)) == '{"platformUpdateDomainCount": 5, "platformFaultDomainCount": 3, "virtualMachines": []}'
    assert json.loads(json.dumps(dict(model))) == model == dict_response

def test_has_no_property():
    dict_response = {
        "platformUpdateDomainCount": 5,
        "platformFaultDomainCount": 3,
        "virtualMachines": [],
        "noprop": "bonjour!"
    }
    model = BasicResource(dict_response)
    assert (
        model.platform_update_domain_count ==
        model["platformUpdateDomainCount"] ==
        dict_response["platformUpdateDomainCount"] ==
        5
    )
    assert not hasattr(model, "no_prop")
    with pytest.raises(AttributeError) as ex:
        model.no_prop

    assert str(ex.value) == "'BasicResource' object has no attribute 'no_prop'"
    assert model["noprop"] == dict_response["noprop"] == "bonjour!"

    # let's add it to model now

    class BasicResourceWithProperty(BasicResource):
        no_prop: str = rest_field(name="noprop")

    model = BasicResourceWithProperty(
        platform_update_domain_count=5,
        platform_fault_domain_count=3,
        virtual_machines=[],
        no_prop="bonjour!"
    )
    assert (
        model.no_prop ==
        model["noprop"] ==
        dict_response["noprop"] ==
        "bonjour!"
    )

def test_original_and_attr_name_same():

    class MyModel(Model):
        hello: str = rest_field()

        @overload
        def __init__(self, *, hello: str):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    dict_response = {"hello": "nihao"}
    model = MyModel(hello="nihao")
    assert model.hello == model["hello"] == dict_response["hello"]

class OptionalModel(Model):
    optional_str: Optional[str] = rest_field()
    optional_time: Optional[datetime.time] = rest_field()
    optional_dict: Optional[Dict[str, Optional[Pet]]] = rest_field(name="optionalDict")
    optional_model: Optional[Pet] = rest_field()
    optional_myself: Optional["OptionalModel"] = rest_field()

    @overload
    def __init__(
        self,
        *,
        optional_str: Optional[str] = None,
        optional_time: Optional[datetime.time] = None,
        optional_dict: Optional[Dict[str, Optional[Pet]]] = None,
        optional_myself: Optional["OptionalModel"] = rest_field(),
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def test_optional_property():

    dict_response = {
        "optional_str": "hello!",
        "optional_time": None,
        "optionalDict": {
            "Eugene": {
                "name": "Eugene",
                "species": "Dog",
            },
            "Lady": None,
        },
        "optional_model": None,
        "optional_myself": {
            "optional_str": None,
            "optional_time": '11:34:56',
            "optionalDict": None,
            "optional_model": {
                "name": "Lady",
                "species": "Newt"
            },
            "optional_myself": None
        }
    }

    model = OptionalModel(dict_response)
    assert model.optional_str == model["optional_str"] == "hello!"
    assert model.optional_time == model["optional_time"] == None
    assert model.optional_dict == model["optionalDict"] == {
        "Eugene": {
            "name": "Eugene",
            "species": "Dog",
        },
        "Lady": None,
    }
    assert model.optional_dict
    assert model.optional_dict["Eugene"].name == model.optional_dict["Eugene"]["name"] == "Eugene"
    assert model.optional_dict["Lady"] is None

    assert model.optional_myself == model["optional_myself"] == {
        "optional_str": None,
        "optional_time": '11:34:56',
        "optionalDict": None,
        "optional_model": {
            "name": "Lady",
            "species": "Newt"
        },
        "optional_myself": None
    }
    assert model.optional_myself
    assert model.optional_myself.optional_str is None
    assert model.optional_myself.optional_time == datetime.time(11, 34, 56)
    assert model.optional_myself.optional_dict is None
    assert model.optional_myself.optional_model
    assert model.optional_myself.optional_model.name == "Lady"
    assert model.optional_myself.optional_model.species == "Newt"
    assert model.optional_myself.optional_myself is None

def test_modify_dict():
    model = BasicResource(
        platform_update_domain_count=5,
        platform_fault_domain_count=3,
        virtual_machines=[]
    )

    # now let's modify the model as a dict
    model["platformUpdateDomainCount"] = 100
    assert model.platform_update_domain_count == model["platformUpdateDomainCount"] == 100

def test_modify_property():
    dict_response = {
        "platformUpdateDomainCount": 5,
        "platformFaultDomainCount": 3,
        "virtualMachines": []
    }
    model = BasicResource(
        platform_update_domain_count=5,
        platform_fault_domain_count=3,
        virtual_machines=[]
    )

    # now let's modify the model through it's properties
    model.platform_fault_domain_count = 2000
    model['platformFaultDomainCount']
    assert model.platform_fault_domain_count == model["platformFaultDomainCount"] == 2000

def test_property_is_a_type():
    class Fish(Model):
        name: str = rest_field()
        species: Literal['Salmon', 'Halibut'] = rest_field()

        @overload
        def __init__(self, *, name: str, species: Literal['Salmon', 'Halibut']):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Fishery(Model):
        fish: Fish = rest_field()

        @overload
        def __init__(self, *, fish: Fish):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    fishery = Fishery({"fish": {"name": "Benjamin", "species": "Salmon"}})
    assert isinstance(fishery.fish, Fish)
    assert fishery.fish.name == fishery.fish['name'] == fishery['fish']['name'] == "Benjamin"
    assert fishery.fish.species == fishery.fish['species'] == fishery['fish']['species'] == "Salmon"

def test_datetime_deserialization():
    class DatetimeModel(Model):
        datetime_value: datetime.datetime = rest_field(name="datetimeValue")

        @overload
        def __init__(self, *, datetime_value: datetime.datetime):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    val_str = "9999-12-31T23:59:59.999Z"
    val = isodate.parse_datetime(val_str)
    model = DatetimeModel({"datetimeValue": val_str})
    assert model['datetimeValue'] == val_str
    assert model.datetime_value == val

    class BaseModel(Model):
        my_prop: DatetimeModel = rest_field(name="myProp")

    model = BaseModel({"myProp": {"datetimeValue": val_str}})
    assert isinstance(model.my_prop, DatetimeModel)
    model.my_prop['datetimeValue']
    assert model.my_prop['datetimeValue'] == model['myProp']['datetimeValue'] == val_str
    assert model.my_prop.datetime_value == val

def test_date_deserialization():
    class DateModel(Model):
        date_value: datetime.date = rest_field(name="dateValue")

        @overload
        def __init__(self, *, date_value: datetime.date):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    val_str = "2016-02-29"
    val = isodate.parse_date(val_str)
    model = DateModel({"dateValue": val_str})
    assert model['dateValue'] == val_str
    assert model.date_value == val

    class BaseModel(Model):
        my_prop: DateModel = rest_field(name="myProp")

    model = BaseModel({"myProp": {"dateValue": val_str}})
    assert isinstance(model.my_prop, DateModel)
    assert model.my_prop['dateValue'] == model['myProp']['dateValue'] == val_str
    assert model.my_prop.date_value == val

def test_time_deserialization():
    class TimeModel(Model):
        time_value: datetime.time = rest_field(name="timeValue")

        @overload
        def __init__(self, *, time_value: datetime.time):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    val_str = '11:34:56'
    val = datetime.time(11, 34, 56)
    model = TimeModel({"timeValue": val_str})
    assert model['timeValue'] == val_str
    assert model.time_value == val

    class BaseModel(Model):
        my_prop: TimeModel = rest_field(name="myProp")

    model = BaseModel({"myProp": {"timeValue": val_str}})
    assert isinstance(model.my_prop, TimeModel)
    assert model.my_prop['timeValue'] == model['myProp']['timeValue'] == val_str
    assert model.my_prop.time_value == val

class SimpleRecursiveModel(Model):
    name: str = rest_field()
    me: "SimpleRecursiveModel" = rest_field()

    @overload
    def __init__(self, *, name: str, me: "SimpleRecursiveModel"):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def test_model_recursion():

    dict_response = {
        "name": "Snoopy",
        "me": {
            "name": "Egg",
            "me": {
                "name": "Chicken"
            }
        }
    }

    model = SimpleRecursiveModel(dict_response)
    assert model['name'] == model.name == "Snoopy"
    assert model['me'] == {
        "name": "Egg",
        "me": {
            "name": "Chicken"
        }
    }
    assert isinstance(model.me, SimpleRecursiveModel)
    assert model.me['name'] == model.me.name == "Egg"
    assert model.me['me'] == {"name": "Chicken"}
    assert model.me.me.name == "Chicken"

def test_dictionary_deserialization():
    class DictionaryModel(Model):
        prop: Dict[str, datetime.datetime] = rest_field()

        @overload
        def __init__(self, *, prop: datetime.datetime):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    val_str = "9999-12-31T23:59:59.999Z"
    val = isodate.parse_datetime(val_str)
    dict_response = {
        "prop": {
            "datetime": val_str
        }
    }
    model = DictionaryModel(dict_response)
    assert model['prop'] == {"datetime": val_str}
    assert model.prop == {"datetime": val}

def test_attr_and_rest_case():

    class ModelTest(Model):
        our_attr: str = rest_field(name="ourAttr")

        @overload
        def __init__(self, *, our_attr: str):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    test_model = ModelTest({"ourAttr": "camel"})
    assert test_model.our_attr == test_model["ourAttr"] == "camel"

    test_model = ModelTest(ModelTest({"ourAttr": "camel"}))
    assert test_model.our_attr == test_model["ourAttr"] == "camel"

    test_model = ModelTest(our_attr="snake")
    assert test_model.our_attr == test_model['ourAttr'] == "snake"

def test_dictionary_deserialization_model():

    class DictionaryModel(Model):
        prop: Dict[str, Pet] = rest_field()

        @overload
        def __init__(self, *, prop: Dict[str, Pet]):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    dict_response = {
        "prop": {
            "Eugene": {
                "name": "Eugene",
                "species": "Dog",
            },
            "Lady": {
                "name": "Lady",
                "species": "Newt",
            }
        }
    }

    model = DictionaryModel(dict_response)
    assert model['prop'] == {
        "Eugene": {
            "name": "Eugene",
            "species": "Dog",
        },
        "Lady": {
            "name": "Lady",
            "species": "Newt",
        }
    }
    assert model.prop == {
        "Eugene": Pet({"name": "Eugene", "species": "Dog"}),
        "Lady": Pet({"name": "Lady", "species": "Newt"})
    }
    assert model.prop["Eugene"].name == model.prop["Eugene"]["name"] == "Eugene"
    assert model.prop["Eugene"].species == model.prop["Eugene"]["species"] == "Dog"
    assert model.prop["Lady"].name == model.prop["Lady"]["name"] == "Lady"
    assert model.prop["Lady"].species == model.prop["Lady"]["species"] == "Newt"

def test_list_deserialization():
    class ListModel(Model):
        prop: List[datetime.datetime] = rest_field()

        @overload
        def __init__(self, *, prop: List[datetime.datetime]):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    val_str = "9999-12-31T23:59:59.999Z"
    val = isodate.parse_datetime(val_str)
    dict_response = {
        "prop": [val_str, val_str]
    }
    model = ListModel(dict_response)
    assert model['prop'] == [val_str, val_str]
    assert model.prop == [val, val]

def test_list_deserialization_model():
    class ListModel(Model):
        prop: List[Pet] = rest_field()

        @overload
        def __init__(self, *, prop: List[Pet]):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    dict_response = {
        "prop": [
            {"name": "Eugene", "species": "Dog"},
            {"name": "Lady", "species": "Newt"}
        ]
    }
    model = ListModel(dict_response)
    assert model["prop"] == [
        {"name": "Eugene", "species": "Dog"},
        {"name": "Lady", "species": "Newt"}
    ]
    assert model.prop == [
        Pet({"name": "Eugene", "species": "Dog"}),
        Pet({"name": "Lady", "species": "Newt"})
    ]
    assert len(model.prop) == 2
    assert model.prop[0].name == model.prop[0]["name"] == "Eugene"
    assert model.prop[0].species == model.prop[0]["species"] == "Dog"
    assert model.prop[1].name == model.prop[1]["name"] == "Lady"
    assert model.prop[1].species == model.prop[1]["species"] == "Newt"

def test_set_deserialization():
    class SetModel(Model):
        prop: Set[datetime.datetime] = rest_field()

        @overload
        def __init__(self, *, prop: Set[datetime.datetime]):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    val_str = "9999-12-31T23:59:59.999Z"
    val = isodate.parse_datetime(val_str)
    dict_response = {
        "prop": set([val_str, val_str])
    }
    model = SetModel(dict_response)
    assert model['prop'] == set([val_str, val_str])
    assert model.prop == set([val, val])

def test_tuple_deserialization():
    class TupleModel(Model):
        prop: Tuple[str, datetime.datetime] = rest_field()

        @overload
        def __init__(self, *, prop: Tuple[str, datetime.datetime]):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    val_str = "9999-12-31T23:59:59.999Z"
    val = isodate.parse_datetime(val_str)
    dict_response = {
        "prop": (val_str, val_str)
    }
    model = TupleModel(dict_response)
    assert model['prop'] == (val_str, val_str)
    assert model.prop == (val_str, val)

def test_list_of_tuple_deserialization_model():

    class Owner(Model):
        name: str = rest_field()
        pet: Pet = rest_field()

        @overload
        def __init__(self, *, name: str, pet: Pet):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class ListOfTupleModel(Model):
        prop: List[Tuple[Pet, Owner]] = rest_field()

        @overload
        def __init__(self, *, prop: List[Tuple[Pet, Owner]]):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    eugene = {"name": "Eugene", "species": "Dog"}
    lady = {"name": "Lady", "species": "Newt"}
    giacamo = {"name": "Giacamo", "pet": eugene}
    elizabeth = {"name": "Elizabeth", "pet": lady}

    dict_response: Dict[str, Any] = {
        "prop": [(eugene, giacamo), (lady, elizabeth)]
    }
    model = ListOfTupleModel(dict_response)
    assert (
        model['prop'] ==
        model.prop ==
        [(eugene, giacamo), (lady, elizabeth)] ==
        [(Pet(eugene), Owner(giacamo)), (Pet(lady), Owner(elizabeth))]
    )
    assert len(model.prop[0]) == len(model['prop'][0]) == 2
    assert model.prop[0][0].name == model.prop[0][0]['name'] == "Eugene"
    assert model.prop[0][0].species == model.prop[0][0]['species'] == "Dog"
    assert model.prop[0][1].name == "Giacamo"
    assert model.prop[0][1].pet == model.prop[0][0]
    assert model.prop[0][1].pet.name == model.prop[0][1]["pet"]["name"] == "Eugene"
    assert model.prop[1][0] == model.prop[1][1].pet

class RecursiveModel(Model):
    name: str = rest_field()
    list_of_me: Optional[List["RecursiveModel"]] = rest_field(name="listOfMe")
    dict_of_me: Optional[Dict[str, "RecursiveModel"]] = rest_field(name="dictOfMe")
    dict_of_list_of_me: Optional[Dict[str, List["RecursiveModel"]]] = rest_field(name="dictOfListOfMe")
    list_of_dict_of_me: Optional[List[Dict[str, "RecursiveModel"]]] = rest_field(name="listOfDictOfMe")

    @overload
    def __init__(
        self,
        *,
        name: str,
        list_of_me: Optional[List["RecursiveModel"]] = None,
        dict_of_me: Optional[Dict[str, "RecursiveModel"]] = None,
        dict_of_list_of_me: Optional[Dict[str, List["RecursiveModel"]]] = None,
        list_of_dict_of_me: Optional[List[Dict[str, "RecursiveModel"]]] = None,
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def test_model_recursion_complex():

    dict_response = {
        "name": "it's me!",
        "listOfMe": [
            {
                "name": "it's me!",
                "listOfMe": None,
                "dictOfMe": None,
                "dictOfListOfMe": None,
                "listOfDictOfMe": None,
            }
        ],
        "dictOfMe": {
            "me": {
                "name": "it's me!",
                "listOfMe": None,
                "dictOfMe": None,
                "dictOfListOfMe": None,
                "listOfDictOfMe": None,
            }
        },
        "dictOfListOfMe": {
            "many mes": [
                {
                    "name": "it's me!",
                    "listOfMe": None,
                    "dictOfMe": None,
                    "dictOfListOfMe": None,
                    "listOfDictOfMe": None,
                }
            ]
        },
        "listOfDictOfMe": [
            {"me": {
                    "name": "it's me!",
                    "listOfMe": None,
                    "dictOfMe": None,
                    "dictOfListOfMe": None,
                    "listOfDictOfMe": None,
                }
            }
        ]
    }

    model = RecursiveModel(dict_response)
    assert model.name == model['name'] == "it's me!"
    assert model['listOfMe'] == [
        {
            "name": "it's me!",
            "listOfMe": None,
            "dictOfMe": None,
            "dictOfListOfMe": None,
            "listOfDictOfMe": None,
        }
    ]
    assert model.list_of_me == [RecursiveModel({
        "name": "it's me!",
        "listOfMe": None,
        "dictOfMe": None,
        "dictOfListOfMe": None,
        "listOfDictOfMe": None,
    })]
    assert model.list_of_me
    assert model.list_of_me[0].name == "it's me!"
    assert model.list_of_me[0].list_of_me is None
    assert isinstance(model.list_of_me, List)
    assert isinstance(model.list_of_me[0], RecursiveModel)

    assert model['dictOfMe'] == {
        "me": {
            "name": "it's me!",
            "listOfMe": None,
            "dictOfMe": None,
            "dictOfListOfMe": None,
            "listOfDictOfMe": None,
        }
    }
    assert model.dict_of_me == {"me": RecursiveModel({
        "name": "it's me!",
        "listOfMe": None,
        "dictOfMe": None,
        "dictOfListOfMe": None,
        "listOfDictOfMe": None,
    })}

    assert isinstance(model.dict_of_me, Dict)
    assert isinstance(model.dict_of_me["me"], RecursiveModel)

    assert model['dictOfListOfMe'] == {
        "many mes": [
            {
                "name": "it's me!",
                "listOfMe": None,
                "dictOfMe": None,
                "dictOfListOfMe": None,
                "listOfDictOfMe": None,
            }
        ]
    }
    assert model.dict_of_list_of_me == {
        "many mes": [
            RecursiveModel({
                "name": "it's me!",
                "listOfMe": None,
                "dictOfMe": None,
                "dictOfListOfMe": None,
                "listOfDictOfMe": None,
            })
        ]
    }
    assert isinstance(model.dict_of_list_of_me, Dict)
    assert isinstance(model.dict_of_list_of_me["many mes"], List)
    assert isinstance(model.dict_of_list_of_me["many mes"][0], RecursiveModel)

    assert model['listOfDictOfMe'] == [
        {"me": {
                "name": "it's me!",
                "listOfMe": None,
                "dictOfMe": None,
                "dictOfListOfMe": None,
                "listOfDictOfMe": None,
            }
        }
    ]
    assert model.list_of_dict_of_me == [{
        "me": RecursiveModel({
            "name": "it's me!",
            "listOfMe": None,
            "dictOfMe": None,
            "dictOfListOfMe": None,
            "listOfDictOfMe": None,
        })
    }]
    assert isinstance(model.list_of_dict_of_me, List)
    assert isinstance(model.list_of_dict_of_me[0], Dict)
    assert isinstance(model.list_of_dict_of_me[0]["me"], RecursiveModel)

    assert json.loads(json.dumps(dict(model))) == model == dict_response

def test_literals():

    class LiteralModel(Model):
        species: Literal["Mongose", "Eagle", "Penguin"] = rest_field()
        age: Literal[1, 2, 3] = rest_field()

        @overload
        def __init__(self, *, species: Literal["Mongose", "Eagle", "Penguin"], age: Literal[1, 2, 3]):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    dict_response = {
        "species": "Mongoose",
        "age": 3
    }
    model = LiteralModel(dict_response)
    assert model.species == model["species"] == "Mongoose"
    assert model.age == model["age"] == 3

    dict_response = {
        "species": "invalid",
        "age": 5
    }
    model = LiteralModel(dict_response)
    assert model["species"] == "invalid"
    assert model["age"] == 5

    assert model.species == "invalid"

    assert model.age == 5

def test_deserialization_callback_override():

    def _callback(obj):
        return [str(entry) for entry in obj]

    class MyModel(Model):
        prop: Sequence[int] = rest_field()

        @overload
        def __init__(self, *, prop: Sequence[int]):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    model_without_callback = MyModel(prop=[1.3, 2.4, 3.5])
    assert model_without_callback.prop == [1.3, 2.4, 3.5]
    assert model_without_callback['prop'] == [1.3, 2.4, 3.5]

    class MyModel(Model):
        prop: Sequence[int] = rest_field(type=_callback)

        @overload
        def __init__(self, *, prop: Any):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    model_with_callback = MyModel(prop=[1.3, 2.4, 3.5])
    assert model_with_callback.prop == ["1.3", "2.4", "3.5"]
    # since the deserialize function is not roundtrippable, once we deserialize
    # the serialized version is the same
    assert model_with_callback['prop'] == [1.3, 2.4, 3.5]

def test_deserialization_callback_override_parent():

    class ParentNoCallback(Model):
        prop: Sequence[float] = rest_field()

        @overload
        def __init__(self, *, prop: Sequence[float]):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def _callback(obj):
        return set([str(entry) for entry in obj])

    class ChildWithCallback(ParentNoCallback):
        prop: Sequence[float] = rest_field(type=_callback)

        @overload
        def __init__(self, *, prop: Sequence[float]):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    parent_model = ParentNoCallback(prop=[1, 1, 2, 3])
    assert parent_model.prop == parent_model["prop"] == [1, 1, 2, 3]

    child_model = ChildWithCallback(prop=[1, 1, 2, 3])
    assert child_model.prop == set(["1", "1", "2", "3"])
    assert child_model['prop'] == [1, 1, 2, 3]

def test_inheritance_basic():
    def _callback(obj):
        return [str(e) for e in obj]

    class Parent(Model):
        parent_prop: List[int] = rest_field(name="parentProp", type=_callback)
        prop: str = rest_field()

        @overload
        def __init__(self, *, parent_prop: List[int], prop: str):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Child(Parent):
        pass

    c = Child(parent_prop=[1, 2, 3], prop="hello")
    assert c == {"parentProp": [1, 2, 3], "prop": "hello"}
    assert c.parent_prop == ["1", "2", "3"]
    assert c.prop == "hello"


class ParentA(Model):
    prop: int = rest_field()

    @overload
    def __init__(self, *, prop: Any):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ParentB(ParentA):
    prop: str = rest_field()
    bcd_prop: Optional[List["ParentB"]] = rest_field(name="bcdProp")

    @overload
    def __init__(self, *, prop: Any, bcd_prop: Optional[List["ParentB"]] = None):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ParentC(ParentB):
    prop: float = rest_field()
    cd_prop: ParentA = rest_field(name="cdProp")

    @overload
    def __init__(
        self,
        *,
        prop: Any,
        bcd_prop: List[ParentB],
        cd_prop: ParentA,
        **kwargs
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ChildD(ParentC):
    d_prop: Tuple[ParentA, ParentB, ParentC, Optional["ChildD"]] = rest_field(name="dProp")

    @overload
    def __init__(
        self,
        *,
        prop: Any,
        bcd_prop: List[ParentB],
        cd_prop: ParentA,
        d_prop: Tuple[ParentA, ParentB, ParentC, Optional["ChildD"]],
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def test_model_dict_comparisons():
    class Inner(Model):
        prop: str = rest_field()

        @overload
        def __init__(
            self,
            *,
            prop: str,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)


    class Outer(Model):
        inner: Inner = rest_field()

        @overload
        def __init__(
            self,
            *,
            inner: Inner,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def _tests(outer):
        assert outer.inner.prop == outer["inner"].prop == outer.inner["prop"] == outer["inner"]["prop"] == "hello"
        assert outer.inner == outer["inner"] == {"prop": "hello"}
        assert outer == {"inner": {"prop": "hello"}}
    _tests(Outer(inner=Inner(prop="hello")))
    _tests(Outer({"inner": {"prop": "hello"}}))

def test_model_dict_comparisons_list():
    class Inner(Model):
        prop: str = rest_field()

        @overload
        def __init__(
            self,
            *,
            prop: str,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)


    class Outer(Model):
        inner: List[Inner] = rest_field()

        @overload
        def __init__(
            self,
            *,
            inner: List[Inner],
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def _tests(outer):
        assert outer.inner[0].prop == outer["inner"][0].prop == outer.inner[0]["prop"] == outer["inner"][0]["prop"] == "hello"
        assert outer.inner == outer["inner"] == [{"prop": "hello"}]
        assert outer == {"inner": [{"prop": "hello"}]}

    _tests(Outer(inner=[Inner(prop="hello")]))
    _tests(Outer({"inner": [{"prop": "hello"}]}))

def test_model_dict_comparisons_dict():
    class Inner(Model):
        prop: str = rest_field()

        @overload
        def __init__(
            self,
            *,
            prop: str,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)


    class Outer(Model):
        inner: Dict[str, Inner] = rest_field()

        @overload
        def __init__(
            self,
            *,
            inner: Dict[str, Inner],
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def _tests(outer):
        assert outer.inner["key"].prop == outer["inner"]["key"].prop == outer.inner["key"]["prop"] == outer["inner"]["key"]["prop"] == "hello"
        assert outer.inner == outer["inner"] == {"key": {"prop": "hello"}}
        with pytest.raises(AttributeError):
            outer.inner.key
        assert outer.inner["key"] == outer["inner"]["key"] == {"prop": "hello"}
        assert outer == {"inner": {"key": {"prop": "hello"}}}

    _tests(Outer(inner={"key": Inner(prop="hello")}))
    _tests(Outer({"inner": {"key": {"prop": "hello"}}}))


def test_inheritance_4_levels():
    a = ParentA(prop=3.4)
    assert a.prop == 3.4
    assert a['prop'] == 3.4
    assert a == {"prop": 3.4}
    assert isinstance(a, Model)

    b = ParentB(prop=3.4, bcd_prop=[ParentB(prop=4.3)])
    assert b.prop == "3.4"
    assert b['prop'] == 3.4
    assert b.bcd_prop == [ParentB(prop=4.3)]
    assert b["bcdProp"] != [{"prop": 4.3, "bcdProp": None}]
    assert b["bcdProp"] == [{"prop": 4.3}]
    assert b.bcd_prop
    assert b.bcd_prop[0].prop == "4.3"
    assert b.bcd_prop[0].bcd_prop is None
    assert b == {"prop": 3.4, "bcdProp": [{"prop": 4.3}]}
    assert isinstance(b, ParentB)
    assert isinstance(b, ParentA)

    c = ParentC(prop=3.4, bcd_prop=[b], cd_prop=a)
    assert c.prop == c['prop'] == 3.4
    assert c.bcd_prop == [b]
    assert c.bcd_prop
    assert isinstance(c.bcd_prop[0], ParentB)
    assert c['bcdProp'] == [b] == [{"prop": 3.4, "bcdProp": [{"prop": 4.3}]}]
    assert c.cd_prop == a
    assert c['cdProp'] == a == {"prop": 3.4}
    assert isinstance(c.cd_prop, ParentA)

    d = ChildD(prop=3.4, bcd_prop=[b], cd_prop=a, d_prop=(a, b, c, ChildD(prop=3.4, bcd_prop=[b], cd_prop=a, d_prop=(a, b, c, None))))
    assert d == {
        'prop': 3.4,
        'bcdProp': [b],
        'cdProp': a,
        'dProp': (
            a, b, c,
            {
                'prop': 3.4,
                'bcdProp': [b],
                'cdProp': a,
                'dProp': (
                    a, b, c, None
                )
            }
        )
    }
    assert d.prop == d['prop'] == 3.4
    assert d.bcd_prop == [b]
    assert d.bcd_prop
    assert isinstance(d.bcd_prop[0], ParentB)
    assert d.cd_prop == a
    assert isinstance(d.cd_prop, ParentA)
    assert d.d_prop[0] == a # at a
    assert isinstance(d.d_prop[0], ParentA)
    assert d.d_prop[1] == b
    assert isinstance(d.d_prop[1], ParentB)
    assert d.d_prop[2] == c
    assert isinstance(d.d_prop[2], ParentC)
    assert isinstance(d.d_prop[3], ChildD)

    assert isinstance(d.d_prop[3].d_prop[0], ParentA)
    assert isinstance(d.d_prop[3].d_prop[1], ParentB)
    assert isinstance(d.d_prop[3].d_prop[2], ParentC)
    assert d.d_prop[3].d_prop[3] is None

def test_multiple_inheritance_basic():
    class ParentOne(Model):
        parent_one_prop: str = rest_field(name="parentOneProp")

        @overload
        def __init__(
            self,
            *,
            parent_one_prop: str,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class ParentTwo(Model):
        parent_two_prop: int = rest_field(name="parentTwoProp", type=lambda x: str(x))

        @overload
        def __init__(
            self,
            *,
            parent_two_prop: int,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Child(ParentOne, ParentTwo):

        @overload
        def __init__(
            self,
            *,
            parent_one_prop: str,
            parent_two_prop: int,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    c = Child(parent_one_prop="Hello", parent_two_prop=3)
    assert c == {"parentOneProp": "Hello", "parentTwoProp": 3}
    assert c.parent_one_prop == "Hello"
    assert c.parent_two_prop == "3"
    assert isinstance(c, Child)
    assert isinstance(c, ParentOne)
    assert isinstance(c, ParentTwo)

def test_multiple_inheritance_mro():
    class A(Model):
        prop: str = rest_field()

        @overload
        def __init__(self, *, prop: str) -> None:
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class B(Model):
        prop: int = rest_field(type=lambda x: int(x))

        @overload
        def __init__(self, *, prop: str) -> None:
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class C(A, B):
        pass

    assert A(prop="1").prop == "1"
    assert B(prop="1").prop == 1
    assert C(prop="1").prop == "1"  # A should take precedence over B

class Feline(Model):
    meows: bool = rest_field()
    hisses: bool = rest_field()
    siblings: Optional[List["Feline"]] = rest_field()

    @overload
    def __init__(
        self,
        *,
        meows: bool,
        hisses: bool,
        siblings: Optional[List["Feline"]] = None,
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Owner(Model):
    first_name: str = rest_field(name="firstName", type=lambda x: x.capitalize())
    last_name: str = rest_field(name="lastName", type=lambda x: x.capitalize())

    @overload
    def __init__(
        self,
        *,
        first_name: str,
        last_name: str,
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class PetModel(Model):
    name: str = rest_field()
    owner: Owner = rest_field()

    @overload
    def __init__(self, *, name: str, owner: Owner):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Cat(PetModel, Feline):
    likes_milk: bool = rest_field(name="likesMilk", type=lambda x: True)

    @overload
    def __init__(
        self,
        *,
        name: str,
        owner: Owner,
        meows: bool,
        hisses: bool,
        likes_milk: bool,
        siblings: Optional[List[Feline]],
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class CuteThing(Model):
    how_cute_am_i: float = rest_field(name="howCuteAmI")

    @overload
    def __init__(self, *, how_cute_am_i: float):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Kitten(Cat, CuteThing):
    eats_mice_yet: bool = rest_field(name="eatsMiceYet")

    @overload
    def __init__(
        self,
        *,
        name: str,
        owner: Owner,
        meows: bool,
        hisses: bool,
        likes_milk: bool,
        siblings: Optional[List[Feline]],
        how_cute_am_i: float,
        eats_mice_yet: bool,
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def test_multiple_inheritance_complex():
    cat = Cat(
        name="Stephanie",
        owner=Owner(first_name="cecil", last_name="cai"),  # gets capitalized in attr
        meows=True,
        hisses=True,
        likes_milk=False,  # likes_milk will change to True on the attribute
        siblings=[Feline(meows=True, hisses=False)]
    )
    assert cat == {
        "name": "Stephanie",
        "owner": {
            "firstName": "cecil",
            "lastName": "cai",
        },
        "meows": True,
        "hisses": True,
        "likesMilk": False,
        "siblings": [{
            "meows": True,
            "hisses": False,
        }]
    }
    assert cat.name == "Stephanie"
    assert isinstance(cat.owner, Owner)
    assert cat.owner.first_name == "Cecil"
    assert cat.owner.last_name == "Cai"
    assert cat.meows
    assert cat.hisses
    assert cat.likes_milk
    assert cat.siblings
    assert len(cat.siblings) == 1
    assert isinstance(cat.siblings[0], Feline)

    kitten = Kitten(
        name="Stephanie",
        owner=Owner(first_name="cecil", last_name="cai"),  # gets capitalized in attr
        meows=True,
        hisses=True,
        likes_milk=False,  # likes_milk will change to True on the attribute
        siblings=[Feline(meows=True, hisses=False)],
        how_cute_am_i=1.0,
        eats_mice_yet=True,
    )
    assert kitten != {
        "name": "Stephanie",
        "owner": {
            "firstName": "cecil",
            "lastName": "cai",
        },
        "meows": True,
        "hisses": True,
        "likesMilk": False,
        "siblings": [{
            "meows": True,
            "hisses": False,
            "siblings": None  # we don't automatically set None here
        }],
        "howCuteAmI": 1.0,
        "eatsMiceYet": True,
    }
    assert kitten == {
        "name": "Stephanie",
        "owner": {
            "firstName": "cecil",
            "lastName": "cai",
        },
        "meows": True,
        "hisses": True,
        "likesMilk": False,
        "siblings": [{
            "meows": True,
            "hisses": False,
        }],
        "howCuteAmI": 1.0,
        "eatsMiceYet": True,
    }
    assert kitten.name == "Stephanie"
    assert isinstance(kitten.owner, Owner)
    assert kitten.owner.first_name == "Cecil"
    assert kitten.owner.last_name == "Cai"
    assert kitten.meows
    assert kitten.hisses
    assert kitten.likes_milk
    assert kitten.siblings
    assert len(kitten.siblings) == 1
    assert isinstance(kitten.siblings[0], Feline)
    assert kitten.eats_mice_yet
    assert kitten.how_cute_am_i == 1.0
    assert isinstance(kitten, PetModel)
    assert isinstance(kitten, Cat)
    assert isinstance(kitten, Feline)
    assert isinstance(kitten, CuteThing)

class A(Model):
    b: "B" = rest_field()

    @overload
    def __init__(self, b: "B"):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class B(Model):
    c: "C" = rest_field()

    @overload
    def __init__(self, *, c: "C"):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class C(Model):
    d: str = rest_field()

    @overload
    def __init__(self, *, d: str):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def test_nested_creation():
    a = A({"b": {"c": {"d": "hello"}}})
    assert isinstance(a["b"], Model)
    assert isinstance(a["b"]["c"], Model)
    assert (
        a["b"]["c"] ==
        a["b"].c ==
        a.b.c
        == {"d": "hello"}
    )

    assert (
        a["b"]["c"]["d"] ==
        a["b"].c.d ==
        a.b["c"].d ==
        a["b"]["c"].d ==
        a["b"].c["d"] ==
        a.b["c"]["d"] ==
        a.b.c.d ==
        "hello"
    )


def test_nested_setting():
    a = A({"b": {"c": {"d": "hello"}}})

    # set with dict
    a["b"]["c"]["d"] = "setwithdict"
    assert (
        a["b"]["c"]["d"] ==
        a["b"].c.d ==
        a.b["c"].d ==
        a["b"]["c"].d ==
        a["b"].c["d"] ==
        a.b["c"]["d"] ==
        a.b.c.d ==
        "setwithdict"
    )

    # set with attr
    a.b.c.d = "setwithattr"
    assert a["b"]["c"]["d"] == "setwithattr"
    assert (
        a["b"]["c"]["d"] ==
        a["b"].c.d ==
        a.b["c"].d ==
        a["b"]["c"].d ==
        a["b"].c["d"] ==
        a.b["c"]["d"] ==
        a.b.c.d ==
        "setwithattr"
    )

class BaseModel(Model):

    inner_model: "InnerModel" = rest_field(name="innerModel")

    @overload
    def __init__(self, *, inner_model: "InnerModel"):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class InnerModel(Model):

    datetime_field: datetime.datetime = rest_field(name="datetimeField")

    @overload
    def __init__(self, *, datetime_field: datetime.datetime):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def test_nested_deserialization():

    serialized_datetime = "9999-12-31T23:59:59.999Z"

    model = BaseModel({"innerModel": {"datetimeField": serialized_datetime}})
    assert model.inner_model["datetimeField"] == model["innerModel"]["datetimeField"] == serialized_datetime
    assert model.inner_model.datetime_field == model["innerModel"].datetime_field == isodate.parse_datetime(serialized_datetime)

    new_serialized_datetime = "2022-12-31T23:59:59.999Z"
    model.inner_model.datetime_field = isodate.parse_datetime(new_serialized_datetime)
    assert model.inner_model["datetimeField"] == '2022-12-31T23:59:59.999000Z'

class X(Model):
    y: "Y" = rest_field()

    @overload
    def __init__(self, *, y: "Y"):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Y(Model):
    z: "Z" = rest_field()

    @overload
    def __init__(self, *, z: "Z"):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Z(Model):
    zval: datetime.datetime = rest_field()

    @overload
    def __init__(self, *, zval: datetime.datetime):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def test_nested_update():
    serialized_datetime = "9999-12-31T23:59:59.999Z"
    parsed_datetime = isodate.parse_datetime(serialized_datetime)
    x = X({"y": {"z": {"zval": serialized_datetime}}})
    assert (
        x.y.z.zval ==
        x["y"].z.zval ==
        x.y["z"].zval ==
        x["y"]["z"].zval ==
        parsed_datetime
    )
    assert (
        x.y.z["zval"] ==
        x.y["z"]["zval"] ==
        x["y"].z["zval"] ==
        x["y"]["z"]["zval"] ==
        serialized_datetime
    )


def test_deserialization_is():
    # test without datetime deserialization
    a = A({"b": {"c": {"d": "hello"}}})
    assert a.b is a.b
    assert a.b.c is a.b.c
    assert a.b.c.d is a.b.c.d


    serialized_datetime = "9999-12-31T23:59:59.999Z"
    x = X({"y": {"z": {"zval": serialized_datetime}}})
    assert x.y is x.y
    assert x.y.z is x.y.z

    assert x.y.z.zval == isodate.parse_datetime(serialized_datetime)

class ModelWithReadonly(Model):
    normal_property: str = rest_field(name="normalProperty")
    readonly_property: str = rest_field(name="readonlyProperty", readonly=True)

    @overload
    def __init__(self, *, normal_property: str,):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# def test_readonly():
#     # we pass the dict to json, so readonly shouldn't show up in the JSON version
#     model = ModelWithReadonly(normal_property="normal_property")
#     model.readonly_property = "set"
#     assert model.readonly_property == "set"
#     assert model["readonlyProperty"] is None

# def test_readonly_roundtrip():
#     # this is the model we get back from the service, we want to send it without readonly
#     received_model = ModelWithReadonly({"normalProperty": "foo", "readonlyProperty": "bar"})
#     assert received_model.normal_property == received_model["normalProperty"] == "foo"
#     assert received_model.readonly_property == received_model["readonlyProperty"] == "bar"

def test_serialization_initialization_and_setting():
    serialized_datetime = "9999-12-31T23:59:59.999000Z"
    parsed_datetime = isodate.parse_datetime(serialized_datetime)

    # pass in parsed
    z = Z(zval=parsed_datetime)
    assert z.zval == parsed_datetime
    assert z["zval"] == serialized_datetime

    # pass in dict
    z = Z({"zval": serialized_datetime})
    assert z.zval == parsed_datetime
    assert z["zval"] == serialized_datetime

    # assert setting
    serialized_datetime = "2022-12-31T23:59:59.999000Z"
    z.zval = isodate.parse_datetime(serialized_datetime)
    assert z["zval"] == serialized_datetime

def test_copy_of_input():
    class TestModel(Model):
        data: List[int] = rest_field()

        @overload
        def __init__(self, *, data: List[int]):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    raw = [1, 2, 3]
    m = TestModel(data=raw)
    assert not m.data is raw
    assert m.data == raw
    raw.append(4)
    assert m.data == [1, 2, 3]

def test_inner_model_custom_serializer():
    class InnerModel(Model):
        prop: str = rest_field(type=lambda x: x[::-1])

        @overload
        def __init__(self, *, prop: str):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class OuterModel(Model):
        innie: InnerModel = rest_field()

        @overload
        def __init__(self, *, innie: InnerModel):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    outer = OuterModel({"innie": {"prop": "hello"}})
    assert outer.innie["prop"] == outer["innie"]["prop"] == "hello"
    assert outer.innie.prop == outer["innie"].prop == "olleh"

def test_default_value():
    class MyModel(Model):
        prop_default_str: str = rest_field(name="propDefaultStr", default="hello")
        prop_optional_str: Optional[str] = rest_field(name="propOptionalStr", default=None)
        prop_default_int: int = rest_field(name="propDefaultInt", default=1)
        prop_optional_int: Optional[int] = rest_field(name="propOptionalInt", default=None)

        @overload
        def __init__(
            self,
            *,
            prop_default_str: str = "hello",
            prop_optional_str: Optional[str] = "propOptionalStr",
            prop_default_int: int = 1,
            prop_optional_int: Optional[int] = None,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    my_model = MyModel()
    assert my_model.prop_default_str == my_model["propDefaultStr"] == "hello"
    assert my_model.prop_optional_str is my_model["propOptionalStr"] is None
    assert my_model.prop_default_int == my_model["propDefaultInt"] == 1
    assert my_model.prop_optional_int is my_model["propOptionalInt"] is None
    assert my_model == {
        "propDefaultStr": "hello",
        "propOptionalStr": None,
        "propDefaultInt": 1,
        "propOptionalInt": None
    }

    my_model = MyModel(prop_default_str="goodbye")
    assert my_model.prop_default_str == my_model["propDefaultStr"] == "goodbye"
    assert my_model.prop_optional_str is my_model["propOptionalStr"] is None
    assert my_model.prop_default_int == my_model["propDefaultInt"] == 1
    assert my_model.prop_optional_int is my_model["propOptionalInt"] is None
    assert my_model == {
        "propDefaultStr": "goodbye",
        "propOptionalStr": None,
        "propDefaultInt": 1,
        "propOptionalInt": None
    }

    my_model = MyModel(prop_optional_int=4)
    assert my_model.prop_default_str == my_model["propDefaultStr"] == "hello"
    assert my_model.prop_optional_str is my_model["propOptionalStr"] is None
    assert my_model.prop_default_int == my_model["propDefaultInt"] == 1
    assert my_model.prop_optional_int == my_model["propOptionalInt"] == 4
    assert my_model == {
        "propDefaultStr": "hello",
        "propOptionalStr": None,
        "propDefaultInt": 1,
        "propOptionalInt": 4
    }

    my_model = MyModel({"propDefaultInt": 5})
    assert my_model.prop_default_str == my_model["propDefaultStr"] == "hello"
    assert my_model.prop_optional_str is my_model["propOptionalStr"] is None
    assert my_model.prop_default_int == my_model["propDefaultInt"] == 5
    assert my_model.prop_optional_int is my_model["propOptionalInt"] is None
    assert my_model == {
        "propDefaultStr": "hello",
        "propOptionalStr": None,
        "propDefaultInt": 5,
        "propOptionalInt": None
    }

def test_pass_models_in_dict():
    class Inner(Model):
        str_property: str = rest_field(name="strProperty")

        @overload
        def __init__(
            self,
            *,
            str_property: str,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Outer(Model):
        inner_property: Inner = rest_field(name="innerProperty")

        @overload
        def __init__(
            self,
            *,
            inner_property: Inner,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def _tests(model: Outer):
        assert (
            {"innerProperty": {"strProperty": "hello"}} ==
            {"innerProperty": Inner(str_property="hello")} ==
            {"innerProperty": Inner({"strProperty": "hello"})} ==
            Outer(inner_property=Inner(str_property="hello")) ==
            Outer(inner_property=Inner({"strProperty": "hello"})) ==
            Outer({"innerProperty": {"strProperty": "hello"}}) ==
            Outer({"innerProperty": Inner(str_property="hello")}) ==
            Outer({"innerProperty": Inner({"strProperty": "hello"})}) ==
            model
        )
    _tests(Outer(inner_property=Inner(str_property="hello")))
    _tests(Outer(inner_property=Inner({"strProperty": "hello"})))
    _tests(Outer({"innerProperty": {"strProperty": "hello"}}))
    _tests(Outer({"innerProperty": Inner(str_property="hello")}))
    _tests(Outer({"innerProperty": Inner({"strProperty": "hello"})}))

def test_mutability_list():
    class Inner(Model):
        str_property: str = rest_field(name="strProperty")

        @overload
        def __init__(
            self,
            *,
            str_property: str,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Middle(Model):
        inner_property: List[Inner] = rest_field(name="innerProperty")
        prop: str = rest_field()

        @overload
        def __init__(
            self,
            *,
            inner_property: List[Inner],
            prop: str,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Outer(Model):
        middle_property: Middle = rest_field(name="middleProperty")

        @overload
        def __init__(
            self,
            *,
            middle_property: Model,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    original_dict = {"middleProperty": {"innerProperty": [{"strProperty": "hello"}], "prop": "original"}}
    model = Outer(original_dict)
    assert model is not original_dict

    # set with dict syntax
    assert model.middle_property is model["middleProperty"]
    middle_property = model.middle_property
    middle_property["prop"] = "new"
    assert model["middleProperty"] is model.middle_property is middle_property
    assert model["middleProperty"]["prop"] == model.middle_property.prop == "new"

    # set with attr syntax
    middle_property.prop = "newest"
    assert model["middleProperty"] is model.middle_property is middle_property
    assert model["middleProperty"]["prop"] == model.middle_property.prop == "newest"

    # modify innerproperty list
    assert (
        model["middleProperty"]["innerProperty"][0] is
        model.middle_property.inner_property[0]
    )
    assert (
        model["middleProperty"]["innerProperty"][0] is
        model.middle_property["innerProperty"][0] is
        model["middleProperty"].inner_property[0] is
        model.middle_property.inner_property[0]
    )
    inner_property = model["middleProperty"]["innerProperty"][0]

    # set with dict syntax
    inner_property["strProperty"] = "nihao"
    assert (
        model["middleProperty"]["innerProperty"][0] is
        model.middle_property["innerProperty"][0] is
        model["middleProperty"].inner_property[0] is
        model.middle_property.inner_property[0]
    )
    assert (
        model["middleProperty"]["innerProperty"][0]["strProperty"] ==
        model.middle_property["innerProperty"][0]["strProperty"] ==
        model["middleProperty"].inner_property[0]["strProperty"] ==
        model.middle_property.inner_property[0]["strProperty"] ==
        model["middleProperty"]["innerProperty"][0].str_property ==
        model.middle_property["innerProperty"][0].str_property ==
        model["middleProperty"].inner_property[0].str_property ==
        model.middle_property.inner_property[0].str_property ==
        "nihao"
    )

def test_mutability_dict():
    class Inner(Model):
        str_property: str = rest_field(name="strProperty")

        @overload
        def __init__(
            self,
            *,
            str_property: str,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Middle(Model):
        inner_property: Dict[str, Inner] = rest_field(name="innerProperty")
        prop: str = rest_field()

        @overload
        def __init__(
            self,
            *,
            inner_property: Dict[str, Inner],
            prop: str,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Outer(Model):
        middle_property: Middle = rest_field(name="middleProperty")

        @overload
        def __init__(
            self,
            *,
            middle_property: Model,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    original_dict = {"middleProperty": {"innerProperty": {"inner": {"strProperty": "hello"}}, "prop": "original"}}
    model = Outer(original_dict)
    assert model is not original_dict

    # set with dict syntax
    assert model.middle_property is model["middleProperty"]
    middle_property = model.middle_property
    middle_property["prop"] = "new"
    assert model["middleProperty"] is model.middle_property is middle_property
    assert model["middleProperty"]["prop"] == model.middle_property.prop == "new"

    # set with attr syntax
    middle_property.prop = "newest"
    assert model["middleProperty"] is model.middle_property is middle_property
    assert model["middleProperty"]["prop"] == model.middle_property.prop == "newest"

    # modify innerproperty list
    assert (
        model["middleProperty"]["innerProperty"]["inner"] is
        model.middle_property.inner_property["inner"]
    )
    assert (
        model["middleProperty"]["innerProperty"]["inner"] is
        model.middle_property["innerProperty"]["inner"] is
        model["middleProperty"].inner_property["inner"] is
        model.middle_property.inner_property["inner"]
    )
    inner_property = model["middleProperty"]["innerProperty"]["inner"]

    # set with dict syntax
    inner_property["strProperty"] = "nihao"
    assert (
        model["middleProperty"]["innerProperty"]["inner"] is
        model.middle_property["innerProperty"]["inner"] is
        model["middleProperty"].inner_property["inner"] is
        model.middle_property.inner_property["inner"]
    )
    assert (
        model["middleProperty"]["innerProperty"]["inner"]["strProperty"] ==
        model.middle_property["innerProperty"]["inner"]["strProperty"] ==
        model["middleProperty"].inner_property["inner"]["strProperty"] ==
        model.middle_property.inner_property["inner"]["strProperty"] ==
        model["middleProperty"]["innerProperty"]["inner"].str_property ==
        model.middle_property["innerProperty"]["inner"].str_property ==
        model["middleProperty"].inner_property["inner"].str_property ==
        model.middle_property.inner_property["inner"].str_property ==
        "nihao"
    )

def test_del_model():
    class TestModel(Model):
        x: Optional[int] = rest_field()

    my_dict = {}
    my_dict["x"] = None

    assert my_dict['x'] is None

    my_model = TestModel({})
    my_model["x"] = None

    assert my_model["x"] is my_model.x is None

    my_model = TestModel({"x": 7})
    my_model.x = None

    assert "x" not in my_model
    assert my_model.x is None

    with pytest.raises(KeyError):
        del my_model["x"]
    my_model.x = 8

    del my_model["x"]
    assert "x" not in my_model
    assert my_model.x is my_model.get("x") is None

    with pytest.raises(AttributeError):
        del my_model.x
    my_model.x = None
    assert "x" not in my_model
    assert my_model.x is my_model.get("x") is None

def test_pop_model():
    ...


##### REWRITE BODY COMPLEX INTO THIS FILE #####

def test_complex_basic():
    class Basic(Model):
        id: Optional[int] = rest_field(default=None)
        name: Optional[str] = rest_field(default=None)
        color: Optional[Literal["cyan", "Magenta", "YELLOW", "blacK"]] = rest_field(default=None)

        @overload
        def __init__(
            self,
            *,
            id: Optional[int] = None,
            name: Optional[str] = None,
            color: Optional[Literal["cyan", "Magenta", "YELLOW", "blacK"]] = None,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    basic = Basic(id=2, name='abc', color="Magenta")
    assert basic == {"id": 2, "name": "abc", "color": "Magenta"}

    basic.id = 3
    basic.name = "new_name"
    basic.color = "blacK"

    assert basic == {"id": 3, "name": "new_name", "color": "blacK"}

    basic["id"] = 4
    basic["name"] = "newest_name"
    basic["color"] = "YELLOW"

    assert basic == {"id": 4, "name": "newest_name", "color": "YELLOW"}


def test_complex_boolean_wrapper():
    class BooleanWrapper(Model):
        field_true: Optional[bool] = rest_field(default=None)
        field_false: Optional[bool] = rest_field(default=None)

        @overload
        def __init__(
            self,
            *,
            field_true: Optional[bool] = None,
            field_false: Optional[bool] = None,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    bool_model = BooleanWrapper(field_true=True, field_false=False)
    assert bool_model == {"field_true": True, "field_false": False}
    bool_model.field_true = False
    bool_model.field_false = True
    assert bool_model == {"field_true": False, "field_false": True}

    bool_model["field_true"] = True
    bool_model["field_false"] = False
    assert bool_model == {"field_true": True, "field_false": False}


def test_complex_byte_wrapper():
    class ByteWrapper(Model):
        field: Optional[bytes] = rest_field(default=None)

        @overload
        def __init__(
            self,
            *,
            field: Optional[bytes] = None,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    byte_string = b'\xff\xfe\xfd\xfc\x00\xfa\xf9\xf8\xf7\xf6'
    mod = ByteWrapper(field=byte_string)
    decoded = base64.b64encode(byte_string).decode()

    def _tests(mod: ByteWrapper):
        assert mod == {"field": decoded}
        assert mod.field == mod["field"] == decoded

    _tests(mod)
    mod.field = byte_string
    _tests(mod)  # test after setting to byte string

    mod["field"] = byte_string
    assert mod["field"] == byte_string
    assert mod.field == decoded

def test_complex_byte_array_wrapper():
    class ByteArrayWrapper(Model):
        field: Optional[bytearray] = rest_field(default=None)

        @overload
        def __init__(
            self,
            *,
            field: Optional[bytearray] = None,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    byte_array = bytearray([0x0FF, 0x0FE, 0x0FD, 0x0FC, 0x000, 0x0FA, 0x0F9, 0x0F8, 0x0F7, 0x0F6])
    decoded = base64.b64encode(byte_array).decode()
    def _tests(model: ByteArrayWrapper):
        assert model == {"field": decoded}
        assert model.field == model["field"] == decoded
    _tests(ByteArrayWrapper(field=byte_array))
    _tests(ByteArrayWrapper({"field": decoded}))

def test_complex_datetimerfc1123_wrapper():
    class Datetimerfc1123Wrapper(Model):
        field: Optional[datetime.datetime] = rest_field(default=None)
        now: Optional[datetime.datetime] = rest_field(default=None)

        @overload
        def __init__(
            self,
            *,
            field: Optional[datetime.datetime] = None,
            now: Optional[datetime.datetime] = None,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    field = "0001-01-01T00:00:00Z"
    now = "2015-05-18T11:38:00Z"

    def _tests(model: Datetimerfc1123Wrapper):
        assert model == {"field": field, "now": now}

        assert model.field == isodate.parse_datetime(field)
        assert model.now == isodate.parse_datetime(now)

        model["field"] = isodate.parse_datetime(field)
        model["now"] = isodate.parse_datetime(now)

        assert model["field"] == isodate.parse_datetime(field)
        assert model["now"] == isodate.parse_datetime(now)

    _tests(Datetimerfc1123Wrapper(field=isodate.parse_datetime(field), now=isodate.parse_datetime(now)))
    _tests(Datetimerfc1123Wrapper({"field": field, "now": now}))


def test_complex_datetime_wrapper():
    class DatetimeWrapper(Model):
        field: datetime.datetime = rest_field(default=None)
        now: datetime.datetime = rest_field(default=None)

        @overload
        def __init__(
            self,
            *,
            field: Optional[datetime.datetime] = None,
            now: Optional[datetime.datetime] = None,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    field = "0001-01-01T00:00:00Z"
    now = "2015-05-18T18:38:00Z"
    def _tests(model: DatetimeWrapper):
        assert model["field"] == field
        assert model["now"] == now
        assert model.field == isodate.parse_datetime(field)
        assert model.now == isodate.parse_datetime(now)
    _tests(DatetimeWrapper(field=isodate.parse_datetime(field), now=isodate.parse_datetime(now)))
    _tests(DatetimeWrapper({"field": field, "now": now}))


def test_complex_date_wrapper():
    class DateWrapper(Model):
        field: datetime.date = rest_field(default=None)
        leap: datetime.date = rest_field(default=None)

        @overload
        def __init__(
            self,
            *,
            field: Optional[datetime.date] = None,
            leap: Optional[datetime.date] = None,
        ):
            ...

        @overload
        def __init__(self, mapping: Mapping[str, Any], /):
            ...

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    field = "0001-01-01"
    leap = "2016-02-29"

    def _tests(model: DateWrapper):
        assert model.field == isodate.parse_date(field)
        assert model["field"] == field

        assert model.leap == isodate.parse_date(leap)
        assert model["leap"] == leap

        model.field = isodate.parse_date(leap)
        assert model.field == isodate.parse_date(leap)
        assert model["field"] == leap

        model["field"] = field
        assert model.field == isodate.parse_date(field)
        assert model["field"] == field

    _tests(DateWrapper({"field": field, "leap": leap}))
    _tests(DateWrapper(field=isodate.parse_date(field), leap=isodate.parse_date(leap)))

class DictionaryWrapper(Model):
    default_program: Dict[str, str] = rest_field(name="defaultProgram", default=None)

    @overload
    def __init__(
        self,
        *,
        default_program: Optional[Dict[str, str]] = None,
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

default_program = {'txt':'notepad', 'bmp':'mspaint', 'xls':'excel', 'exe':'', '': None}

@pytest.mark.parametrize("model", [DictionaryWrapper({"defaultProgram": default_program}), DictionaryWrapper(default_program=default_program)])
def test_complex_dictionary_wrapper(model: DictionaryWrapper):
    assert model == {"defaultProgram": default_program}
    assert model.default_program == model["defaultProgram"] == default_program

@pytest.mark.parametrize("model", [DictionaryWrapper({"defaultProgram": {}}), DictionaryWrapper(default_program={})])
def test_complex_dictionary_wrapper_empty(model: DictionaryWrapper):
    assert model == {"defaultProgram": {}}
    assert model.default_program == model["defaultProgram"] == {}

@pytest.mark.parametrize("model", [DictionaryWrapper({"defaultProgram": None}), DictionaryWrapper(default_program=None)])
def test_complex_dictionary_wrapper_none(model: DictionaryWrapper):
    assert model == {"defaultProgram": None}
    assert model.default_program is None

class ArrayWrapper(Model):
    array: Optional[List[str]] = rest_field(default=None)

    @overload
    def __init__(
        self,
        *,
        array: Optional[List[str]] = None,
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

array_value = ["1, 2, 3, 4", "", None, "&S#$(*Y", "The quick brown fox jumps over the lazy dog"]

@pytest.mark.parametrize("model", [ArrayWrapper(array=array_value), ArrayWrapper({"array": array_value})])
def test_complex_array_wrapper(model: ArrayWrapper):
    assert model == {"array": array_value}
    assert model.array == model["array"] == array_value

    model.array = None
    with pytest.raises(KeyError):
        model["array"]
    assert model.array is None

    model["array"] = [1, 2, 3, 4, 5]
    assert model.array == ["1", "2", "3", "4", "5"]
    assert model["array"] == [1, 2, 3, 4, 5]

@pytest.mark.parametrize("model", [ArrayWrapper(array=[]), ArrayWrapper({"array": []})])
def test_complex_array_wrapper_empty(model: ArrayWrapper):
    assert model == {"array": []}
    assert model.array == model["array"] == []

    model.array = ["bonjour"]
    assert model.array == model["array"] == ["bonjour"]

@pytest.mark.parametrize("model", [ArrayWrapper(array=None), ArrayWrapper({"array": None})])
def test_complex_array_wrapper_none(model: ArrayWrapper):
    assert model == {"array": None}
    assert model.array is model["array"] is None

    model.array = ["bonjour"]
    assert model.array == model["array"] == ["bonjour"]

class PetComplex(Model):
    id: Optional[int] = rest_field(default=None)
    name: Optional[str] = rest_field(default=None)

    @overload
    def __init__(
        self,
        *,
        id: Optional[int] = None,
        name: Optional[str] = None,
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class DogComplex(PetComplex):
    food: Optional[str] = rest_field(default=None)

    @overload
    def __init__(
        self,
        *,
        id: Optional[int] = None,
        name: Optional[str] = None,
        food: Optional[str] = None,
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CatComplex(PetComplex):
    color: Optional[str] = rest_field(default=None)
    hates: Optional[List[DogComplex]] = rest_field(default=None)

    @overload
    def __init__(
        self,
        *,
        id: Optional[int] = None,
        name: Optional[str] = None,
        food: Optional[str] = None,
        color: Optional[str] = None,
        hates: Optional[List[DogComplex]] = None,
    ):
        ...

    @overload
    def __init__(self, mapping: Mapping[str, Any], /):
        ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@pytest.mark.parametrize(
    "model",
    [
        CatComplex(id=2, name="Siameeee", hates=[
            DogComplex(id=1, name="Potato", food="tomato"),
            DogComplex(id=-1, name="Tomato", food="french fries")
        ]),
        CatComplex(id=2, name="Siameeee", hates=[
            DogComplex(id=1, name="Potato", food="tomato"),
            {"id": -1, "name": "Tomato", "food": "french fries"},
        ]),
        CatComplex(id=2, name="Siameeee", hates=[
            {"id": 1, "name": "Potato", "food": "tomato"},
            {"id": -1, "name": "Tomato", "food": "french fries"},
        ]),
    ]
)
def test_complex_inheritance(model: CatComplex):
    assert model.id == model["id"] == 2
    assert model.name == model["name"] == "Siameeee"
    assert model.hates
    assert model.hates[1] == model["hates"][1] == {"id": -1, "name": "Tomato", "food": "french fries"}
    model["breed"] = "persian"
    model["color"] = "green"
    with pytest.raises(AttributeError):
        model.breed
    assert model == {
        'id': 2,
        'name': "Siameeee",
        'color': "green",
        'breed': "persian",
        'hates': [DogComplex(id=1, name="Potato", food="tomato"),
                DogComplex(id=-1, name="Tomato", food="french fries")]
        }

