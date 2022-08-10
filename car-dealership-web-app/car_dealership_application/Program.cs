using CarDealershipWebApp.Models;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;

namespace CarDealershipWebApp
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var host = CreateHostBuilder(args).Build();

            using (var scope = host.Services.CreateScope())
            {
                var services = scope.ServiceProvider;

                try
                {
                    SeedData.Initialize(services);
                }
                catch (Exception ex)
                {
                    var logger = services.GetRequiredService<ILogger<Program>>();
                    logger.LogError(ex, "An error occurred seeding the DB.");
                }
            }

            host.Run();
        }

        public static IHostBuilder CreateHostBuilder(string[] args) =>
            /*
            1) Common practice in handling app secrets:
            - Never store passwords or other sensitive data in source code.
            - Production secrets shouldn't be used for development or test.
            - Secrets shouldn't be deployed with the app.
            - Production secrets should be accessed through a controlled means like environment variables or Azure Key Vault.
           
            2) Mapping an entire object literal to a POCO
            Configuration["Movies:ServiceApiKey"] through Dependency Injection IConfiguration configuration
            var moviesConfig = Configuration.GetSection("Movies").Get<MovieSettings>();

            3) Definition of Managed identities 
            - An app deployed to Azure can take advantage of Managed identities for Azure resources. A managed identity allows the app to authenticate with Azure Key Vault using Azure AD authentication without credentials (Application ID and Password/Client Secret) stored in the app.

            4) List of dependencies
            - Install-Package Azure.Extensions.AspNetCore.Configuration.Secrets
            - Install-Package Azure.Identity

            5) List of configuration
            - Get Object ID of the web app: Click App service -> Identity -> Enable System Assigned managed identity
            
            - Click Key Vault -> Access Policies -> Add Access Policy -> Secret permissions(Get, List) -> Select principal ->
              Enter app object ID -> Select -> Add
            az keyvault set-policy --name {KEY VAULT NAME} --object-id {OBJECT ID} --secret-permissions get list

            6) List of Azure service
            6.1)) AppService
            - Runtime stack: .NET 5
            - Operating system: Windows
            - Continuous deployment: Disable
            - Application insights: No

            6.2)) Key Vault
            - Pricing tier: Standard
            - Default settings for the rest

            7) 
            Non-hierarchical values: The value for SecretName is obtained with config["SecretName"].
            Hierarchical values (sections): Use : (colon) notation or the GetSection extension method. Use either of these approaches to obtain the configuration value:
            config["Section:SecretName"]
            config.GetSection("Section")["SecretName"]
             */
            // CreateDefaultBuilder calls AddUserSecrets when the EnvironmentName is Development.
            Host.CreateDefaultBuilder(args)
                .ConfigureAppConfiguration((context, config) =>
                {
                    //if (context.HostingEnvironment.IsProduction())
                    //{
                    //    var builtConfig = config.Build();
                    //    var secretClient = new SecretClient(
                    //        new Uri($"https://{builtConfig["KeyVaultName"]}.vault.azure.net/"),
                    //        new DefaultAzureCredential());
                    //    config.AddAzureKeyVault(secretClient, new PrefixKeyVaultSecretManager(builtConfig["VaultPrefix"]));
                    //}
                })
                .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.UseStartup<Startup>();
                });
    }
}
