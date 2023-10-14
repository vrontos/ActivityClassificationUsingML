#!/usr/bin/env python3

import usb.core
import usb.util
import time
import argparse

def get_usb_devices():
    usb_devices = []
    devices = usb.core.find(find_all=True)
    for device in devices:
        vid = hex(device.idVendor)
        pid = hex(device.idProduct)
        usb_devices.append((vid, pid))
    return usb_devices  # Return the list of devices


def send(data):
    device = usb.core.find(idVendor=0x0483, idProduct=0x5740)
    for config in device:
        for intf in config:
            if device.is_kernel_driver_active(intf.bInterfaceNumber):
                try:
                    device.detach_kernel_driver(intf.bInterfaceNumber)
                except usb.core.USBError as e:
                    pass
    if device is None:
        raise ValueError("USB device not found.")

    out_endpoint = None
    in_endpoint = None
    for cfg in device:
        for intf in cfg:
            for ep in intf:
                if usb.util.endpoint_direction(ep.bEndpointAddress) == usb.util.ENDPOINT_OUT:
                    out_endpoint = ep
                elif usb.util.endpoint_direction(ep.bEndpointAddress) == usb.util.ENDPOINT_IN:
                    in_endpoint = ep

    if out_endpoint is None or in_endpoint is None:
        raise ValueError("Both OUT and IN endpoints are required for communication.")
    interface = 0
    usb.util.claim_interface(device, interface)
    device.write(out_endpoint.bEndpointAddress, data)
    response = device.read(in_endpoint.bEndpointAddress, in_endpoint.wMaxPacketSize)
    usb.util.release_interface(device, interface)
    usb.util.dispose_resources(device)


def HtB(hex_string): 
    hex_values = hex_string.split()
    binary_values = []
    for hex_value in hex_values:
        decimal_value = int(hex_value, 16)
        binary_value = bin(decimal_value)[2:].zfill(8)
        binary_values.append(binary_value)
    return binary_values


def BtH(binary_string):
    binary_values = binary_string.split()
    hex_values = []
    for binary_value in binary_values:
        decimal_value = int(binary_value, 2)
        hex_value = hex(decimal_value)[2:].upper().zfill(2)
        hex_values.append(hex_value)
    return hex_values


def list_glue(string_list):
    s=""
    for i in string_list:
        s+=i
    return s


def hex_xor(hex1, hex2):
    int1 = int(hex1, 16)
    int2 = int(hex2, 16)
    xor_result = int1 ^ int2
    hex_result=format(xor_result, '02X')
    return hex_result


def checksum(hex_sequence):
    crc = 0x0000
    hex_values = hex_sequence.split()
    for hex_value in hex_values:
        data = int(hex_value, 16)
        crc ^= data << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
    crc &= 0xFFFF
    crcmodem = format(crc, '04X')
    crc0 = format(crc, '04X')
    return "81" + " " + hex_xor(crc0[:2],'55') + " " + "81" + " " + hex_xor(crc0[2:],'55')


def length_cal(command_str):
    l = 10 + len(command_str.split())
    l = hex(l)[2:]
    l = hex_xor(l,'55')
    return "81" + " " + "55" + " "  + "81"+ " "  + l


def message_gen(data_string):
    m = "F0" + " " + length_cal(data_string)+ " " + checksum(data_string) + " " + data_string + " " + "0F"
    return m


def init():
    command= "00 1E 00"
    send(bytes.fromhex(message_gen(command)))


def stop():
    command= "0C 22"
    send(bytes.fromhex(message_gen(command)))


def get_current_data():
    message = message_gen("08 24 02")
    return bytes.fromhex(message)


def split_string(string, block_size):
    string = ''.join(string.split())
    num_blocks = len(string) // block_size
    blocks = [string[i:i+block_size] for i in range(0, len(string), block_size)]
    out = ' '.join(blocks)
    return out

def update_1point(pw, amp, frequency):
    p1= bin(pw)[2:].zfill(12) + bin(int(2*amp)+300)[2:].zfill(10) + "0"*10
    p2= bin(pw)[2:].zfill(12) + bin(int(2*(-amp))+300)[2:].zfill(10) + "0"*10
    ramp = 3
    block_1_8 = "00000001"
    block_2_8 = bin(1)[2:].zfill(4) + bin(ramp)[2:].zfill(4)
    block_3_16 = bin(((1000*2)//frequency))[2:].zfill(15) + "0" 
    command_block= "04 20" + " " + split_string(list_glue(BtH(split_string(block_1_8 + block_2_8 + block_3_16 + p1 + p2, 8))), 2)
    CB=command_block.split()
    for i in range(len(CB)):
        if CB[i] in ["0F","F0","81","55","0f","f0"]:
            CB[i]="81" + " " + hex_xor(CB[i], '55')
    CB2=''
    for i in CB:
        CB2 = CB2 + i
    command_block=split_string(''.join(CB2.split()), 2)
    ml_message = "F0" +" " + length_cal(command_block) + " " + checksum(command_block) + " " + command_block + " " + "0F"
    return bytes.fromhex(ml_message)


def send_message(pw, amp, frequency, timer):
    message1=update_1point(pw, amp, frequency)
    message2=get_current_data()
    init()
    for _ in range(timer*10):
        send(message1)
        send(message2)
        time.sleep(0.1)
    stop()
    
def main(pw, amp, frequency, timer):
    send_message(pw, amp, frequency, timer)
    usb_devices = get_usb_devices()  # Get the list of devices
    for vid, pid in usb_devices:  # Print the devices
        print("Vendor ID (VID):", vid)
        print("Product ID (PID):", pid)
        print("----------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stimulator script with arguments')
    
    parser.add_argument('pw', type=int, help='Pulse width')
    parser.add_argument('amp', type=float, help='Amplitude')
    parser.add_argument('frequency', type=int, help='Frequency')
    parser.add_argument('timer', type=float, help='Timer')
    
    args = parser.parse_args()
    
    main(args.pw, args.amp, args.frequency, args.timer)

#send_message(120, 20, 50, 2)




