class Db:
    # api to database operation

    @classmethod
    def save_meter_value(cls, datetime, meter_id, temperature):
        print('Meter value to save: ',datetime,meter_id,temperature)

    @classmethod
    def get_meters_info(cls, qr_code):
        stub_meter = (1,(36,-198)) # meter_id, (offset_x,offset_y)
        return [stub_meter]

