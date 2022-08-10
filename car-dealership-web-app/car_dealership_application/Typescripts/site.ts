/// <reference types='bootstrap' />

import { setVisible } from './ui_controller';
import * as bootstrap from 'bootstrap';

export function load() {
    $(document).ready(function () {
        console.log("Running");

        // Initialize bootstrap tooltips
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl)
        });

        // Toggle the side navigation
        // Source: https://startbootstrap.com/template/sb-admin
        const sidebarToggle = document.body.querySelector('#sidebarToggle');
        if (sidebarToggle) {
            // Uncomment Below to persist sidebar toggle between refreshes
            // if (localStorage.getItem('sb|sidebar-toggle') === 'true') {
            //     document.body.classList.toggle('sb-sidenav-toggled');
            // }
            sidebarToggle.addEventListener('click', event => {
                event.preventDefault();
                document.body.classList.toggle('sb-sidenav-toggled');
                localStorage.setItem('sb|sidebar-toggle',
                    JSON.stringify(document.body.classList.contains('sb-sidenav-toggled')));
            });
        }

        console.log("Finsih running");

        setVisible('#page', true);
        setVisible('#loading', false);

        // Register service worker if browser supports
        //if ('serviceWorker' in navigator) {
        //    navigator.serviceWorker.register('/sw.js').then(param => {
        //        setVisible('#page', true);
        //        setVisible('#loading', false);
        //    });
        //}
        //// Else continue as usual
        //else {
        //    setVisible('#page', true);
        //    setVisible('#loading', false);
        //}
    });
}