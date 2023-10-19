#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from kafka import KafkaProducer
import json

# Create a Kafka Producer Instance
producer = KafkaProducer(
    bootstrap_servers = 'localhost:9092',
    value_serializer = lambda v: json.dumps(v).encode('utf-8')
)

