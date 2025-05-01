# f1-data-api
Python based api to return F1 data

If no data is present in local cache, process can take up to 5-10 mins to get the data. Keeping it simple for local use. \n

If I continue building out I would convert this to a websocket (probably the most simple solution)or use some queue polling API (maybe use redis & celery or AWS/similar queue service to avoid long running single connection) In which case I would just have a job id to poll every minute or so.
