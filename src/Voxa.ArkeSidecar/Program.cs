using Voxa.ArkeSidecar;

try
{
    await SidecarApplication.Build(args).RunAsync().ConfigureAwait(false);
}
catch (ArgumentException ex)
{
    Console.Error.WriteLine($"Voxa.ArkeSidecar: {ex.Message}");
    Environment.ExitCode = 2;
}
