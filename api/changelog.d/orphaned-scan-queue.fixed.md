A scan whose worker was killed no longer holds its provider scan queue forever: with orphan task recovery enabled, the scan is marked failed and the next queued scan is dispatched
