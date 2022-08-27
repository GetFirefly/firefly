-module(init).

-export([boot/1]).

% SCTP events which will be subscribed by default upon opening the socket.
% NB: "data_io_event" controls delivery of #sctp_sndrcvinfo{} ancilary
% data, not events (which are normal data) in fact; it may be needed in
% order to get the AssocID of data just received:
%
-record(sctp_event_subscribe,
	{
	  data_io_event,          % true,	% Used by gen_sctp
	  association_event,      % true, 	% Used by gen_sctp
	  address_event,          % true,	% Unlikely to happen...
	  send_failure_event,     % true,	% Delivered as an ERROR
	  peer_error_event,       % true,	% Delivered as an ERROR
	  shutdown_event,         % true,	% Used by gen_sctp
	  partial_delivery_event, % true,	% Unlikely to happen...
	  adaptation_layer_event, % false	% Probably not needed...
	  authentication_event    % false       % Not implemented yet...
	}).

boot(_Arg) ->
    ok.
