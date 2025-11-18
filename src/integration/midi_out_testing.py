import mido

print(mido.get_output_names())
print(mido.get_input_names())

out = mido.open_output('Teensy MIDI:Teensy MIDI MIDI 1 24:0')


# new Send NOTE ON TEST

msg_on_one = mido.Message('note_on', note=60, velocity=100, channel=0)
out.send(msg_on_one)
print("SEND NOTE ON COMPLETE")

# send note OFF test
msg_off_one = mido.Message('note_off', note=60, velocity=100, channel=0)
out.send(msg_off_one)
print("SEND NOTE OFF COMPLETE")

