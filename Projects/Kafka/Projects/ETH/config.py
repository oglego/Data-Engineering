#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# Update the config dict with the values you want to pass 
config = {
            "key" : "insert_api_key",
            "crypto" : "X:ETHUSD",
            "current_date" : str(date.today()),
            "start_date" : str((datetime.now() - relativedelta(years=2)).strftime('%Y-%m-%d')),
            "topic" : "ETH"
    }

