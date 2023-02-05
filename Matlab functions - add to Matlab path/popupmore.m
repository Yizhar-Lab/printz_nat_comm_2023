function popupmore(fh,varargin)
% SUBPLOT POPUP FUNCTION WITH ADDITIONAL ARGUMENTS
%   makes use of the function POPUP and PVPMOD
%   
%   EXAMPLES
%       popupmore(fh)                                                 => popup(gca); standard function
%       popupmore(fh,'lg',???)                                        => same plus legend
%       popupmore(fh,'cm',cmp)                                        => same plus colormap(cmp)
%       popupmore(fh,'cm',cmp,'cb','EastOutside')                     => same plus colorbar in 'EastOutside' location
%       popupmore(fh,'cm',cmp,'cb','EastOutside','lb','label string') => same plus colorbar ylabel(label string)
%   
%       feel free to add options
%   
%   (c) kandler, steffen.kandler[at]bccn.uni-freiburg.de, last updated 29. Apr. 2009


pvpmod(varargin);

if ~isempty(get(fh,'UserData'))
    error('UserData contains data - POPUPMORE can''t continue')
end

% standard with colormap and colorbar and colorbar label
if exist('lb','var')
    set(fh,'UserData',{cm;cb;lb})
    set(fh,'WindowButtonDownFcn','userData = get(gcf,''UserData'');  popup(gca); colormap(userData{1}); cbh = colorbar; set(cbh,''Location'',userData{2}); ylabel(cbh,userData{3},''FontSize'',16,''FontWeight'',''bold''); set(cbh,''TickDir'',''out''); set(cbh,''Box'',''on''); set(cbh,''LineWidth'',2); set(cbh,''FontWeight'',''bold'',''FontSize'',16)')
else
    % standard with colormap and colorbar
    if exist('cb','var')
        set(fh,'UserData',{cm;cb})
        set(fh,'WindowButtonDownFcn','userData = get(gcf,''UserData'');  popup(gca); colormap(userData{1}); cbh = colorbar; set(cbh,''Location'',userData{2}); set(cbh,''TickDir'',''out''); set(cbh,''Box'',''on''); set(cbh,''LineWidth'',2); set(cbh,''FontWeight'',''bold'',''FontSize'',16)')
    else
        % standard with colormap
        if exist('cm','var')
            set(fh,'UserData',{cm})
            set(fh,'WindowButtonDownFcn','userData = get(gcf,''UserData''); popup(gca); colormap(userData{1})')
        else
            % standard with legend
            if exist('lg','var')
                % ???
            else
                % standard popup function
                set(fh,'WindowButtonDownFcn','popup(gca)')
            end
        end
    end
end



% %% pvpmod, modified from UE
% function pvpmod(x)
% 
% if ~isempty(x)
%     if length(x) == 1 & isstruct(x{1})
%         x = x{1};
%         allfields = fieldnames(x);
%         for k = 1:length(allfields)
%             eval(['assignin(''caller'', allfields{k}, x.' allfields{k} ');']);
%         end
%     elseif mod(length(x),2) == 0
%         for k = 1:2:size(x,2)
%             assignin('caller',x{k},x{k+1});
%         end
%     else
%         error('pvpmod error')
%     end
% end
% 
% 
% %% popup, modified from UE
% function popup(ah,fh,varargin)
% 
% marksizegui = 1;
% 
% pvpmod(varargin);
% 
% if strcmp(get(ah,'type'),'axes')
%     if exist('fh')
%         if ischar(fh)
%             a = setgcf(fh);
%         else
%             a = figure(fh);
%         end
%     else
%         a = setgcf('popup figure');
%     end
%     clf
%     bh = copyobj(ah,a);
%     set(bh,'units','normalized','position', [.1 .1 .8 .8]);
%     set(bh,'buttondownfcn','');
% end
% 
% oph = [findobj(a,'label','options');
% findobj(a,'label','&Options')];
% if isempty(oph)
%     oph = uimenu('parent',a,'label','&Options','tag','optionsmenu');
% end
% mph = uimenu('parent',oph,'label','rename figure','tag','rename figure','callback','popsubplotctrl(''renamefig'',gcbf);');
% mph = uimenu('parent', oph,'label', 'adjust axis','tag','adjust axis','callback','popsubplotctrl(''adj_axis'',gcbf);');
% 
% return
