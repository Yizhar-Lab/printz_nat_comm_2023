function [figh, S] = setgcf(figname, action, varargin);
% SETGCF - find/create a named figure
% figh = setgcf(figname, action);
% Looks for/generates a named figure performes the requested action
% and returns the figure handle. If a figure of this name is not 
% found a new one is created and its handle returned.
%
% If several figures with the same name are found the 
% functions is cancelled.
%
% legal values for <action>
% 'create'      creates a new figure with <figname>, keeping others of the same name
% 'clear'       clears the figure
% 'delete'      deletes a figure with <figname>
% 'reset'       see helpwin reset
% 'replace'     re-creates a figure of that name
%
% U. Egert 7/97


%################ MAIN PROGRAM ###################################################


% find the appropriate figure

figh = findobj('name', figname);

switch nargin
case 1
   action = ' ';
end;


if strcmp(action, 'create') | (isempty(figh) & (strcmp(action, ' ') | ~strcmp(action, 'delete')))
   figh = figure;
   set(figh,...
      'Color','white',...
      'PaperType','a4', ...
      'InvertHardcopy','off',...
      'paperorientation','landscape',...
      'numbertitle', 'off', ...
      'name', figname, varargin{:})
   orient landscape
   S = 0;
else
   if length(figh) > 1
      msgbox(['found more than one figure named ' figname ', aborting'])
      return
   end;
   
   % execute the requested action
   switch action
   case 'clear'
      try
         figure(figh)
         clf(figh);
      catch
         figh = figure;
         end;
      S=2;
   case 'delete'
      delete(figh)
      S=-1;
   case 'reset'
      reset(figh);
      set(figh, 'name', figname);
      S=3;
   case 'replace'
      Position = get(figh,'Position');
      delete(figh);
      figh=setgcf(figname,'create','Position',Position);
      S=4;
   case ' '
      figure(figh);
      S = 1;
   end;
   drawnow
%   S = 1;
end;

