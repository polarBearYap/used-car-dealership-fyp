import { FilterCache } from './ui_controller';

export function load() {
	$(document).ready(function () {
		const filterCacheName: string = "leadFilterCache";
		const checkboxes: JQuery<HTMLInputElement> = $('input[name=filtercolumn]');
		const columnCheckBoxes: JQuery<HTMLInputElement> = $('input[type="checkbox"]').not("#selectAll") as JQuery<HTMLInputElement>;

		console.log(filterCacheName);
		console.log(checkboxes);
		console.log(columnCheckBoxes);

		const filterCache = new FilterCache(filterCacheName, checkboxes, columnCheckBoxes);

		filterCache.loadFilterCache();
		filterCache.addEventListeners();
	});
}