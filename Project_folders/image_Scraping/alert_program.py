from twilio.rest import Client
import os

accountSID = os.environ["TWILIO_SID"]
authToken = os.environ["TWILIO_TOKEN"]
myTwilioNumber = os.environ["TWILIO_NUMBER"]
myCellPhone = os.environ["my_number"]


def send_end_alert(project_name, body='Default', accountSID=accountSID,
                   authToken=authToken, myTeilioNumber=myTwilioNumber,
                   myCellPhone=myCellPhone):

    if body != 'Default': body = body
    else: body = 'Your project, {}, has completed!'.format(project_name)

    twilioCli = Client(accountSID, authToken)
    message = twilioCli.messages.create(body=body,
                                        from_=myTwilioNumber,
                                        to=myCellPhone)
    return message

if __name__ == '__main__':
    send_end_alert('test')
