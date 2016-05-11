# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:55:53 2015

@author: freso388
"""


def check_sequence(sequence):

    aa_letters = set(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
    sequence = str(sequence).strip().upper()
    if len(sequence) < 15:
        return "SEQ_TOO_SHORT"
    if all(c in aa_letters for c in sequence):
        return sequence
    else:
        return "SEQ_ERROR"


def check_email(email):

    at_found = False
    dot_found = False

    for letter in email:
        if letter == "@":
            at_found = True
        if letter == ".":
            dot_found = True

    if at_found and dot_found:
        return email
    else:
        return "EMAIL_ERROR"


def vis_prediction(name, prediction, probabilities):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 3))
    plt.plot(range(len(probabilities)), probabilities)
    plt.plot(range(len(prediction)), [0.5] * len(prediction), "r--")
    plt.fill_between(range(len(prediction)),
                     prediction * 1,
                     prediction * 0,
                     color="g", alpha=.3)
    plt.xlabel("Residue")
    plt.ylabel("Protean segment prediction score")
    plt.savefig('vis_' + name + '.pdf', bbox_inches='tight')
    plt.savefig('vis_' + name + '.png', bbox_inches='tight')

    return None


def make_csv(name, sequence, prediction, probabilities):

    from csv import writer
    prediction = [int(pred) for pred in prediction]
    probabilities = ["%.3f" % prob for prob in probabilities]
    with open(name + '.csv', 'wb') as csvfile:
        csv_writer = writer(csvfile, delimiter=' ')
        csv_writer.writerow(sequence)
        csv_writer.writerow(prediction)
        csv_writer.writerow(probabilities)

    return None


def send_mail(send_to, files=None, smtpserver="smtp.gmail.com:587"):

    import smtplib
    from os.path import basename
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.utils import COMMASPACE, formatdate

    assert isinstance(send_to, list)

    send_from = "thrinduilthrowaway@gmail.com"
    subject = "Results from Proteus"
    text = "Thank you for using Proteus."
    login = "thrinduilthrowaway"
    password = "nothing is sacred"

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Subject'] = subject
    msg['Date'] = formatdate(localtime=True)
    msg.attach(MIMEText(text))

    for f in files or []:
        with open(f, "rb") as fil:
            attachment = MIMEApplication(fil.read())
        attachment.add_header('Content-Disposition',
                              'attachment; filename="%s"' % basename(f))
        msg.attach(attachment)

    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login, password)
    server.sendmail(send_from, send_to, msg.as_string())
    server.quit()
