% To plot the overlaid connectivity maps of all cells for one cell type,
% load the corresponding .mat file of that cell type located in this folder,
% and then run the this code.
anat_figure = figure;	subplot(1,2,1); hold on
scatter(dML_spiking_cells_masked(~isConn_spiking_cells_masked),-dDV_spiking_cells_masked(~isConn_spiking_cells_masked),10,[0.7 0.7 0.7],'filled')
scatter(dML_spiking_cells_masked(isConn_spiking_cells_masked),-dDV_spiking_cells_masked(isConn_spiking_cells_masked),30,amp_spiking_cells_masked(isConn_spiking_cells_masked),'filled')
colormap autumn;	C = colorbar;	ylabel(C,'Connection strength (pA)');   C.FontSize = 12;    C.Location = 'southoutside';    caxis(c_lims)
scatter(0,0,30,[0 0 0],'^','filled')
line([grid_edges(1) grid_edges(end)],[0 0],'LineStyle','--','color',[0 0 0])
line([0 0],[grid_edges(1) grid_edges(end)],'LineStyle','--','color',[0 0 0])
xlabel('ML position (\mum; L>0)');   ylabel('DV position (\mum; D>0)');   daspect([1 1 1]);   box off
legend('Non-connected cells','Connected cells','Recorded cell');	title('Cell reference');	axis tight
subplot(1,2,2);	hold on
scatter(spiking_cells_coords_masked(~isConn_spiking_cells_masked,1),spiking_cells_coords_masked(~isConn_spiking_cells_masked,2),10,[0.7 0.7 0.7],'filled')
scatter(spiking_cells_coords_masked(isConn_spiking_cells_masked,1),spiking_cells_coords_masked(isConn_spiking_cells_masked,2),30,amp_spiking_cells_masked(isConn_spiking_cells_masked),'filled')
colormap autumn;	C = colorbar;	ylabel(C,'Connection strength (pA)');   C.FontSize = 12;	C.Location = 'southoutside';    caxis(c_lims)
scatter(recorded_cells_coords(recorded_cells_mask,1),recorded_cells_coords(recorded_cells_mask,2),30,[0 0 0],'^','filled')
xlabel('ML position (\mum)');   ylabel('DV position (\mum)');   set(gca,'YDir','reverse')
ylim([1000 2800]);  xlim([0 800]);  daspect([7 18 1])
legend('Non-connected cells','Connected cells','Recorded cell');	title('Brain reference')
popupmore(anat_figure)
