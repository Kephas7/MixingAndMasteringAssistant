# Signal levels

Internal audio arrays use floating-point amplitude, where full scale is conventionally `-1.0` to `1.0`. Values outside this range can exist during processing but must be controlled before integer export.

Compare processed and original signals at similar perceived loudness. Louder playback can otherwise be mistaken for better processing.
