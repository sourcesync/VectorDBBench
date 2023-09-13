# coding: utf-8

"""
    GSI Floating-Point 32 API

    **Introduction**<br> GSI Technology’s floating-point similarity search API provides an accessible gateway to running searches on GSI’s Gemini® Associative Processing Unit (APU).<br> It works in conjunction with the GSI system management solution which enables users to work with multiple APU boards simultaneously for improved performance.<br><br> **Dataset and Query Format**<br> Dataset embeddings can be in 32- or 64-bit floating point format, and any number of features, e.g. 256 or 512 (there is no upper limit).<br> Query embeddings must have the same floating-point format and number of features as used in the dataset.<br> GSI performs the search and delivers the top-k most similar results.  # noqa: E501

    OpenAPI spec version: 1.0
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six

class GenerateQueriesRequest(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'num_of_records': 'int',
        'num_of_features': 'int',
        'query_type': 'str'
    }

    attribute_map = {
        'num_of_records': 'numOfRecords',
        'num_of_features': 'numOfFeatures',
        'query_type': 'queryType'
    }

    def __init__(self, num_of_records=None, num_of_features=None, query_type='float32'):  # noqa: E501
        """GenerateQueriesRequest - a model defined in Swagger"""  # noqa: E501
        self._num_of_records = None
        self._num_of_features = None
        self._query_type = None
        self.discriminator = None
        self.num_of_records = num_of_records
        self.num_of_features = num_of_features
        if query_type is not None:
            self.query_type = query_type

    @property
    def num_of_records(self):
        """Gets the num_of_records of this GenerateQueriesRequest.  # noqa: E501

        Number of queries to generate  # noqa: E501

        :return: The num_of_records of this GenerateQueriesRequest.  # noqa: E501
        :rtype: int
        """
        return self._num_of_records

    @num_of_records.setter
    def num_of_records(self, num_of_records):
        """Sets the num_of_records of this GenerateQueriesRequest.

        Number of queries to generate  # noqa: E501

        :param num_of_records: The num_of_records of this GenerateQueriesRequest.  # noqa: E501
        :type: int
        """
        if num_of_records is None:
            raise ValueError("Invalid value for `num_of_records`, must not be `None`")  # noqa: E501

        self._num_of_records = num_of_records

    @property
    def num_of_features(self):
        """Gets the num_of_features of this GenerateQueriesRequest.  # noqa: E501

        Number of features of the generated queries  # noqa: E501

        :return: The num_of_features of this GenerateQueriesRequest.  # noqa: E501
        :rtype: int
        """
        return self._num_of_features

    @num_of_features.setter
    def num_of_features(self, num_of_features):
        """Sets the num_of_features of this GenerateQueriesRequest.

        Number of features of the generated queries  # noqa: E501

        :param num_of_features: The num_of_features of this GenerateQueriesRequest.  # noqa: E501
        :type: int
        """
        if num_of_features is None:
            raise ValueError("Invalid value for `num_of_features`, must not be `None`")  # noqa: E501

        self._num_of_features = num_of_features

    @property
    def query_type(self):
        """Gets the query_type of this GenerateQueriesRequest.  # noqa: E501

        Type of query to generate - float32 or uint8  # noqa: E501

        :return: The query_type of this GenerateQueriesRequest.  # noqa: E501
        :rtype: str
        """
        return self._query_type

    @query_type.setter
    def query_type(self, query_type):
        """Sets the query_type of this GenerateQueriesRequest.

        Type of query to generate - float32 or uint8  # noqa: E501

        :param query_type: The query_type of this GenerateQueriesRequest.  # noqa: E501
        :type: str
        """
        allowed_values = ["float32", "uint8"]  # noqa: E501
        if query_type not in allowed_values:
            raise ValueError(
                "Invalid value for `query_type` ({0}), must be one of {1}"  # noqa: E501
                .format(query_type, allowed_values)
            )

        self._query_type = query_type

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(GenerateQueriesRequest, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, GenerateQueriesRequest):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
