#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from kafka import KafkaConsumer
import json
from config import config

# Topic name is obtained from the parameter in the config file
topic_name = config["topic"]

# Create a Kafka Consumer Instance
consumer = KafkaConsumer(
     topic_name,
     bootstrap_servers=['localhost:9092'],
     auto_offset_reset='earliest',
     group_id=None,
     value_deserializer=lambda x: json.loads(x.decode('utf-8')))

