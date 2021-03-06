# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tabml/protos/feature_manager.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tabml.protos import path_pb2 as tabml_dot_protos_dot_path__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tabml/protos/feature_manager.proto',
  package='tabml.protos',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\"tabml/protos/feature_manager.proto\x12\x0ctabml.protos\x1a\x17tabml/protos/path.proto\"?\n\x0b\x42\x61seFeature\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\"\n\x05\x64type\x18\x02 \x02(\x0e\x32\x13.tabml.protos.DType\"t\n\x13TransformingFeature\x12\r\n\x05index\x18\x01 \x02(\x03\x12\x0c\n\x04name\x18\x02 \x02(\t\x12\x14\n\x0c\x64\x65pendencies\x18\x03 \x03(\t\x12*\n\x05\x64type\x18\x04 \x01(\x0e\x32\x13.tabml.protos.DType:\x06STRING\"\xc3\x01\n\rFeatureConfig\x12(\n\x0craw_data_dir\x18\x01 \x01(\x0b\x32\x12.tabml.protos.Path\x12\x14\n\x0c\x64\x61taset_name\x18\x02 \x01(\t\x12\x30\n\rbase_features\x18\x03 \x03(\x0b\x32\x19.tabml.protos.BaseFeature\x12@\n\x15transforming_features\x18\x04 \x03(\x0b\x32!.tabml.protos.TransformingFeature*l\n\x05\x44Type\x12\x08\n\x04\x42OOL\x10\x01\x12\t\n\x05INT32\x10\x02\x12\t\n\x05INT64\x10\x03\x12\n\n\x06STRING\x10\x04\x12\t\n\x05\x46LOAT\x10\x05\x12\n\n\x06\x44OUBLE\x10\x06\x12\x08\n\x04\x44\x41TE\x10\x07\x12\x08\n\x04TIME\x10\x08\x12\x0c\n\x08\x44\x41TETIME\x10\t'
  ,
  dependencies=[tabml_dot_protos_dot_path__pb2.DESCRIPTOR,])

_DTYPE = _descriptor.EnumDescriptor(
  name='DType',
  full_name='tabml.protos.DType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='BOOL', index=0, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INT32', index=1, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INT64', index=2, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='STRING', index=3, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FLOAT', index=4, number=5,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='DOUBLE', index=5, number=6,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='DATE', index=6, number=7,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='TIME', index=7, number=8,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='DATETIME', index=8, number=9,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=458,
  serialized_end=566,
)
_sym_db.RegisterEnumDescriptor(_DTYPE)

DType = enum_type_wrapper.EnumTypeWrapper(_DTYPE)
BOOL = 1
INT32 = 2
INT64 = 3
STRING = 4
FLOAT = 5
DOUBLE = 6
DATE = 7
TIME = 8
DATETIME = 9



_BASEFEATURE = _descriptor.Descriptor(
  name='BaseFeature',
  full_name='tabml.protos.BaseFeature',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='tabml.protos.BaseFeature.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dtype', full_name='tabml.protos.BaseFeature.dtype', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=77,
  serialized_end=140,
)


_TRANSFORMINGFEATURE = _descriptor.Descriptor(
  name='TransformingFeature',
  full_name='tabml.protos.TransformingFeature',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='index', full_name='tabml.protos.TransformingFeature.index', index=0,
      number=1, type=3, cpp_type=2, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='name', full_name='tabml.protos.TransformingFeature.name', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dependencies', full_name='tabml.protos.TransformingFeature.dependencies', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dtype', full_name='tabml.protos.TransformingFeature.dtype', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=4,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=142,
  serialized_end=258,
)


_FEATURECONFIG = _descriptor.Descriptor(
  name='FeatureConfig',
  full_name='tabml.protos.FeatureConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='raw_data_dir', full_name='tabml.protos.FeatureConfig.raw_data_dir', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dataset_name', full_name='tabml.protos.FeatureConfig.dataset_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='base_features', full_name='tabml.protos.FeatureConfig.base_features', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='transforming_features', full_name='tabml.protos.FeatureConfig.transforming_features', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=261,
  serialized_end=456,
)

_BASEFEATURE.fields_by_name['dtype'].enum_type = _DTYPE
_TRANSFORMINGFEATURE.fields_by_name['dtype'].enum_type = _DTYPE
_FEATURECONFIG.fields_by_name['raw_data_dir'].message_type = tabml_dot_protos_dot_path__pb2._PATH
_FEATURECONFIG.fields_by_name['base_features'].message_type = _BASEFEATURE
_FEATURECONFIG.fields_by_name['transforming_features'].message_type = _TRANSFORMINGFEATURE
DESCRIPTOR.message_types_by_name['BaseFeature'] = _BASEFEATURE
DESCRIPTOR.message_types_by_name['TransformingFeature'] = _TRANSFORMINGFEATURE
DESCRIPTOR.message_types_by_name['FeatureConfig'] = _FEATURECONFIG
DESCRIPTOR.enum_types_by_name['DType'] = _DTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BaseFeature = _reflection.GeneratedProtocolMessageType('BaseFeature', (_message.Message,), {
  'DESCRIPTOR' : _BASEFEATURE,
  '__module__' : 'tabml.protos.feature_manager_pb2'
  # @@protoc_insertion_point(class_scope:tabml.protos.BaseFeature)
  })
_sym_db.RegisterMessage(BaseFeature)

TransformingFeature = _reflection.GeneratedProtocolMessageType('TransformingFeature', (_message.Message,), {
  'DESCRIPTOR' : _TRANSFORMINGFEATURE,
  '__module__' : 'tabml.protos.feature_manager_pb2'
  # @@protoc_insertion_point(class_scope:tabml.protos.TransformingFeature)
  })
_sym_db.RegisterMessage(TransformingFeature)

FeatureConfig = _reflection.GeneratedProtocolMessageType('FeatureConfig', (_message.Message,), {
  'DESCRIPTOR' : _FEATURECONFIG,
  '__module__' : 'tabml.protos.feature_manager_pb2'
  # @@protoc_insertion_point(class_scope:tabml.protos.FeatureConfig)
  })
_sym_db.RegisterMessage(FeatureConfig)


# @@protoc_insertion_point(module_scope)
